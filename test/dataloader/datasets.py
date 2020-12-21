import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
import PIL
from PIL import Image
import requests
import warnings
import pickle
from google.cloud import storage
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import imgaug as ia
import skimage.draw
import cv2
import six
from termcolor import colored
import webdataset as wds
from webdataset.filters import batched, shuffle
from webdataset.dataset import warn_and_continue
import io

# try:
#     import torch_xla.core.xla_model as xm
# except ModuleNotFoundError:
#     print('torch-xla not found... continuing with CUDA pytorch.')

def split_by_at(text, by, at):
    """Split a text string into two parts by `by` at position `at`.

    :param text: str
    :param by: str
    :param at: int
    :return:
    """
    text_list = text.split(by)
    left = by.join(text_list[:at])
    right = by.join(text_list[at:])

    return left, right


warnings.filterwarnings(
    "ignore", "\bEXIF\b", UserWarning
)  # ignoring EXIF data in

DEBUG_MODE = 0

if DEBUG_MODE:
    from pyinstrument import Profiler

    profiler = Profiler()


class OpenImagesTransform(object):
    """Joint image, mask transform for OpenImages for FCHash nets.

    """

    def __init__(self, base_aug, fmap_scales=None):
        """
        :param fmap_scales: list of ints; default = [32, 16, 8]
        """
        if fmap_scales is None:
            fmap_scales = [32, 16, 8]
        self.fmap_scales = fmap_scales
        self.base_aug = base_aug

    def __call__(self, image, mask):
        """

        :param image: np.ndarray of shape [H, W, 3] NOTE channel order!
        :param mask: np.ndarray of shape [H, W], dtype=int32
        :return: image, mask
            image: np.array of shape [3, H_out, W_out]
            targets: dict with keys = e.g. [32, 16, 8] corresponding to feature map size
                e.g. mask_aug[fmap_size].shape = [num_categories, H, W]
        """

        # Do albumentations base augs:
        data_aug = self.base_aug(image=image, mask=mask.astype(np.uint16))
        image = data_aug['image']
        mask = data_aug['mask']

        # Generate correct size masks:
        size_augmented = image.shape[0]
        masks = dict()
        for fmap_scale in self.fmap_scales:
            size = int(size_augmented / fmap_scale)
            # NN interp important because otherwise incorrect boundary classes:
            masks[size] = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

        # swap channels from numpy to torch:
        image = swap_channels(image, fwd=False).astype(np.float32)  # [C, H, W]

        return image, masks


class DeepfashionTransform(object):
    """`albumentations` based joint transformation module for deepfashion2 images and targets
    NOTE: will sample only *one* style to be segmented in the targets to enable fast vectorized similarity loss!
    """

    def __init__(self, base_aug, fmap_scales=None):
        """

        :param base_aug: albumentations transforms
        :param fmap_scales: list of ints; default = [32, 16, 8]
        """
        if fmap_scales is None:
            fmap_scales = [32, 16, 8]
        self.fmap_scales = fmap_scales
        self.base_aug = base_aug

    def __call__(self, image, target):
        """

        :param image: np.ndarray of shape [H, W, 3] NOTE channel order!
        :param target: a dict (see deepfashion2 metadata)
        :return: image_aug, targets_aug
            image_aug: np.array of shape [3, H_out, W_out]
            targets_aug: dict with keys = e.g. [32, 16, 8] corresponding to feature map size
                e.g. targets_aug[fmap_size]['segmentation_cat'].shape = [num_categories, H, W]
        """

        # profiler.start()

        # 1) get keypoints + item, poly index:
        segmentation_array = self.get_segmentation_array(target)

        # 2) perform base augmentation to image, keypoints
        # TODO: some polygon points will be outside crop. Deal with in polygons_to_segmap
        data_aug = self.base_aug(image=image, keypoints=segmentation_array[:, 2:])
        image_aug = data_aug['image']
        polygons_aug = np.array(data_aug['keypoints'])  # [n_points, 2]
        segmentation_array[:, 2:] = np.array(polygons_aug)  # replace with unaugmented segmentations

        # 3) generate segmaps for all keypoints, using also index:
        targets_aug = dict()
        size_augmented = image_aug.shape[0]  # TODO: assuming square shape for now!!!
        for fmap_scale in self.fmap_scales:
            size = int(size_augmented / fmap_scale)
            # Initialize segmentation maps:
            segmentation_cat = np.zeros((size, size), dtype=np.int32)  # int valued
            # segmentation_sca = np.zeros((4, size, size), dtype=np.uint8)
            # segmentation_occ = np.zeros((4, size, size), dtype=np.uint8)
            # segmentation_zoo = np.zeros((4, size, size), dtype=np.uint8)
            # segmentation_vie = np.zeros((4, size, size), dtype=np.uint8)  # 1 means not worn, so important!
            segmentation_sty = np.zeros((size, size), dtype=np.int32)

            # Add all item data to segmaps:
            style_used = 0
            style_count = 0
            for key in target.keys():
                if 'item' in key:
                    # Get correct item/polygon idx... yes fucking hacky but shiiiit
                    item_id = eval(key[-1])
                    item_idx = segmentation_array[:, 0] == item_id
                    segmentations_item = segmentation_array[item_idx]

                    # Generate segmentation map out of poly keypoints:
                    segmap = polygons_to_segmap(segmentations_item, size, size_augmented)  # [H, H]; bool

                    category_id = target[key]["category_id"]
                    style = target[key]["style"]
                    # scale = target[key]["scale"]
                    # occlusion = target[key]["occlusion"]
                    # zoom_in = target[key]["zoom_in"]
                    # viewpoint = target[key]["viewpoint"]

                    # Add segmentations to correct category etc:
                    # segmentation_cat[category_id] = segmap
                    segmentation_cat[segmap > 0] = category_id
                    # segmentation_cat[0] = ~segmap  # FFFUuu mistake...
                    # segmentation_sca[scale] = segmap
                    # segmentation_sca[0] = ~segmap
                    # segmentation_occ[occlusion] = segmap
                    # segmentation_occ[0] = ~segmap
                    # segmentation_zoo[zoom_in] = segmap
                    # segmentation_zoo[0] = ~segmap
                    # segmentation_vie[viewpoint] = segmap
                    # segmentation_vie[0] = ~segmap

                    # Add style info:
                    segmentation_sty_this = np.zeros((size, size), dtype=np.int32)
                    segmentation_sty_this[segmap > 0] = style
                    # If nonzero style segmentation map, possibly add as segmentation:
                    # (need to sample uniformly from all existing styles)
                    if segmentation_sty_this.sum() > 0:  # checking if nonzero style segmentation map
                        style_count += 1
                        replace_prob = 1 / style_count
                        if np.random.rand() < replace_prob:  # replace with this style
                            style_used = style
                            segmentation_sty = segmentation_sty_this

            # Pack different size segmaps to dict:
            output_dict = dict(segmentation_cat=segmentation_cat,
                               # segmentation_sca=segmentation_sca,
                               # segmentation_occ=segmentation_occ,
                               # segmentation_zoo=segmentation_zoo,
                               # segmentation_vie=segmentation_vie,
                               segmentation_sty=segmentation_sty,
                               style=style_used)

            targets_aug[size] = output_dict

        # swap channels from numpy to torch:
        image_aug = swap_channels(image_aug, fwd=False).astype(np.float32)  # [C, H, W]

        # profiler.stop()
        # print(profiler.output_text(unicode=True, color=True, show_all=False))
        # raise KeyboardInterrupt

        return image_aug, targets_aug

    def get_segmentation_array(self, target):
        """Generate a single numpy array out of different items and polygons per item. Will be collected into a numpy
        array of shape [n_points, 4], where the columns are item_id (int > 0), poly_id (int >= 0), x, y.

        :param target: dict
        :return poly_array: float, shape [n_points, 4]
        """

        poly_arr_proto = np.array([], dtype=[('item_id', 'U10'), ('poly_id', 'i4'), ('x', 'f4'), ('y', 'f4')])

        poly_array_list = list()
        item_keys = np.sort([key for key in target.keys() if 'item' in key])
        for item_id, item in enumerate(item_keys):
            polys_item = target[item]['segmentation']
            for poly_id, poly in enumerate(polys_item):
                poly = np.array(poly).reshape((-1, 2))
                n = len(poly)
                poly_array_this = np.stack((n * [item_id + 1], n * [poly_id], poly[:, 0], poly[:, 1]), axis=1)
                poly_array_list.append(poly_array_this)

        poly_array = np.concatenate(poly_array_list, axis=0)

        return poly_array


def polygons_to_segmap(segmentations_item, size, size_augmented):
    """Generate a (size, size) shaped segmentation map out of `polygons_`

    :param polygons_: PolygonsOnImage instance
    :param size: int; note ASSUMING this is the actual correct size! Maybe re-think...
    :return:
    """
    # Compute scaling factor:
    scale = size / size_augmented  # to be used when scaling the segmentation keypoints

    result = np.zeros((size, size), dtype=np.float32)
    poly_ids = np.unique(segmentations_item[:, 1])
    for poly_id in poly_ids:
        poly_idx = segmentations_item[:, 1] == poly_id
        poly_this = scale * segmentations_item[poly_idx, 2:]
        y, x = np.split(poly_this, [1], axis=1)
        # Generate polygon pixels:
        rr, cc = skimage.draw.polygon(
            x - .5, y - .5, shape=(size, size))
        # Write to result array:
        if len(rr) > 0:
            result[rr, cc] = 1

    return result.astype(bool)



def load_numpy_from_gcs(bucket_name, blob_name):
    """Remember to close!

    :param bucket:
    :param blob:
    :return: NpzFile for .npz file *or* numpy array for .npy files
    """
    bytes = load_bytes_from_gcs(bucket_name, blob_name)
    stream = io.BytesIO(bytes)
    numpy_data = np.load(stream)
    return numpy_data


def load_pkl_from_gcs(bucket_name, blob_name):
    blob_bytes = load_bytes_from_gcs(bucket_name, blob_name)
    pkl_data = pickle.loads(blob_bytes, fix_imports=True, encoding="ASCII", errors="strict")
    return pkl_data


def load_bytes_from_gcs(bucket_name, blob_name):
    """Remember to close!

    :param bucket:
    :param blob:
    :return: BytesIO stream
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    assert blob.exists(), f'Blob {blob_name} does not exist!'
    blob_bytes = blob.download_as_string()

    return blob_bytes


def load_jpg_from_gcs(bucket_name, blob_name):
    """Loads a JPG from GCS and returns it as a numpy array of shape [H, W, 3].

    :param bucket_name:
    :param blob_name:
    :return:
    """
    blob_bytes = load_bytes_from_gcs(bucket_name, blob_name)
    stream = io.BytesIO(blob_bytes)
    img = np.array(Image.open(stream).convert('RGB'))

    return img


def get_imagenet_paths_and_targets(bucket_name='imidatasets', prefix='imagenet/train/'):
    """List all blobs in folder, collect all paths and extract wnids as targets.

    :param bucket_name:
    :param prefix:
    :return:
    """

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, max_results=2000000)
    paths = list()
    targets = list()
    for blob in blobs:
        if blob.name.split(".")[-1].lower() in ["jpeg", "jpg", "png", "gif"]:
            paths.append(blob.name)
            img_folders = os.path.split(os.path.dirname(blob.name))
            img_folders = [fldr for fldr in img_folders if len(fldr) > 0]  # get rid of artefacts
            if len(img_folders) > 1:
                target = img_folders[-1]
            else:
                target = "None"
            targets.append(target)
    paths = np.array(paths).astype(np.string_)
    targets = np.array(targets).astype(np.string_)

    return paths, targets


class URLDataset(Dataset):
    """Generic dataset for any jpeg URLs.

    """

    def __init__(self,
                 path_list=None,
                 target_list=None,
                 transform=None,
                 target_transform=None,
                 joint_transform=None,
                 cache_path=None,
                 debug_mode=False,
                use_torchvision=False):  # TODO: stupid hack
        """Custom PyTorch dataset for a list/array of URLs.

        :param path_list: list or numpy array of `path`s where path is of form 'https://xyz.jpg'
        :param target_list: list, numpy array or dict of targets corresponding to each path in `path_list`; ndim >= 1 as
            for `path_list`
        :param transform:
        :param target_transform:
        :param joint_transform: joint transform for both image and target containing e.g. keypoints, segmaps, ...
        :param cache_data: whether or not cache data locally to enable faster read access once entire dataset has
                been cached.
        :param
        """

        self.path_list = path_list
        self.target_list = target_list
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.cache_path = cache_path
        self.debug_mode = debug_mode
        self.use_torchvision = use_torchvision
        if debug_mode:
            print(colored('WARNING: RUNNING DATALOADER IN DEBUG MODE!!', 'red'))

        os.makedirs(self.cache_path, exist_ok=True) if self.cache_path is not None else None

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):

        url = self.path_list[index]  # e.g. 'https://*.jpg'
        target = self.target_list[index] if self.target_list is not None else None

        img = self.get_img(url)

        if img is None:
            return None, None

        if self.transform is not None:
            if self.use_torchvision:
                img = self.transform(img)
            else:
                img = np.array(img)

                img = self.transform(image=img)['image']
                img = img[None]

        if self.target_transform is not None:
            target = self.target_transform(target)  # TODO: possibly handle None

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        return img, target

    def get_img(self, url_, convert_to_rgb=True):
        # TODO: this could use some refactoring...

        if self.debug_mode:  # just generate a random "image"
            img = np.random.randn(256, 256, 3).astype(np.uint8)
            img = Image.fromarray(img).convert("RGB")
            return img

        if isinstance(url_, bytes):
            url_ = str(url_, "utf-8")

        _, filename = split_by_at(url_, "/", -1)
        ext = filename.split(".")[-1].lower()

        if self.cache_path is not None:
            cache_file_path = join(self.cache_path, filename)
            cached_file_is_present = os.path.exists(cache_file_path)
        else:
            cache_file_path = None
            cached_file_is_present = False

        if not cached_file_is_present:  # from blob storage if no local fname_img_
            try:
                img_bytes = requests.get(url_).content
                stream = io.BytesIO(img_bytes)
            except Exception as e:  # TODO: handle exceptions
                print(
                    "url:",
                    url_,
                    "filename:",
                    filename,
                )
                print("**** MISSING BLOB:   " + url_)
                print(e)
                return None  # collate_fn will handle missing examples
        else:
            stream = None

        try:
            if ext in ["jpeg", "jpg", "png", "gif"]:  # TODO: not cool
                if stream is not None:
                    img = Image.open(stream)
                else:
                    img = Image.open(cache_file_path)  # TODO: ughh not elegant...
                img = img.convert("RGB") if convert_to_rgb else img.convert("I")  # ensure RGB *BEFORE* saving
                if self.cache_path is not None and stream is not None:  # save if using cache
                    img.save(cache_file_path)

            else:
                print("**** CANNOT HANDLE FILETYPE:   " + url_)
                stream.close() if stream is not None else None
                return None  # collate_fn will handle missing examples

            stream.close() if stream is not None else None

        except OSError as err:
            print(f"Exception while attempting to read image: {err}...", end='')
            try:
                os.remove(cache_file_path)
                print(' deleted!')
            except Exception as err2:
                print(f' could not remove: {err2}')
            stream.close() if stream is not None else None
            return None  # collate_fn will handle missing examples

        # return np.array(img)  # somehow not converted to numpy for *streamed* png files, hence this
        return img  # TODO: return numpy array when using albumentations

    def get_target(self, blob_path_):

        if isinstance(blob_path_, bytes):
            blob_path_ = str(blob_path_, "utf-8")

        local_path, filename = split_by_at(blob_path_, "/", -1)
        if self.cache_path is not None:
            cache_folder = join(self.cache_path, local_path)
            cache_file_path = join(cache_folder, filename)
            os.makedirs(cache_folder, exist_ok=True) if self.cache_path is not None else None
            cached_file_is_present = os.path.exists(cache_file_path)
        else:
            cache_file_path = None
            cached_file_is_present = False

        ext = filename.split(".")[-1].lower()

        if not cached_file_is_present:  # from blob storage if no local file
            try:
                blob = self.bucket.blob(blob_path_)
                assert blob.exists(), 'Blob does not exist!'
                ann = eval(str(blob.download_as_string(), encoding='utf-8'))
            except Exception as e:  # TODO: handle exceptions
                print(
                    "cache_file_path:",
                    cache_file_path,
                    "blob_path_:",
                    blob_path_,
                    "local_path:",
                    local_path,
                    "filename:",
                    filename,
                )
                print("**** MISSING BLOB:   " + blob_path_)
                print(e)
                return None  # collate_fn will handle missing examples
            # Save locally:
            if self.cache_path is not None:
                json.dump(ann, open(cache_file_path, 'w'))
        else:
            ann = json.load(open(cache_file_path, 'r'))

        return ann


class DiskDataset(Dataset):
    """Dataset class for on-disk datasets.

    """

    def __init__(self,
                 root_path='/mnt/disks/datasets',
                 paths_image=None,
                 paths_annotation=None,
                 targets=None,
                 transform=None,
                 target_transform=None,
                 joint_transform=None):
        """Custom dataset for on disk datasets.

        :param paths_image: list or numpy array of `path`s where path is of form 'folder/.../image_filename.jpg'
        :param paths_annotation: list or numpy array of `path`s to a json file, where path is of form
            'folder/.../annotation_filename.json'
        :param targets: numpy array of targets - alternative to `paths_annotation` for simpler targets
        :param transform:
        :param target_transform:
        :param joint_transform: joint transform for both image and target containing e.g. keypoints, segmaps, ...
        :param
        """
        self.root_path = root_path
        self.paths_image = paths_image
        self.paths_annotation = paths_annotation
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

    def __len__(self):
        len_dataset = len(self.paths_image)
        assert len_dataset == len(self.paths_annotation), 'Oops mismatch in image and annotation metadata...'
        return len_dataset

    def __getitem__(self, index):

        path_img = self.paths_image[index]  # e.g. 'imagenet/train/xxxx.jpg'
        img = self.get_img(path_img)

        if self.paths_annotation is not None:
            path_ann = self.paths_annotation[index]
            target = self.get_target(path_ann)
        else:
            target = self.targets[index]

        if img is None:  # let collate_fn handle missing examples
            return None, None

        if self.transform is not None:
            img = self.transform(image=img)['image']  # albumentations only for now

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        return img, target

    def get_img(self, path_img_):
        """

        :param path_img_: e.g. 'youtube-bb-full/images/v_AAB6lO-XiKE.mp4.238000.jpg'
        :return:
        """
        if isinstance(path_img_, bytes):
            path_img_ = str(path_img_, "utf-8")

        full_path = join(self.root_path, path_img_)
        img = Image.open(full_path).convert('RGB')

        return np.array(img)

    def get_target(self, path_ann_):
        """Load a JSON annotation file from disk.

        :param path_ann_:  e.g. 'youtube-bb-full/annotations/v_AAB6lO-XiKE.mp4.238000.json'
        :return:
        """
        if path_ann_ is None:
            return None

        if isinstance(path_ann_, bytes):
            path_ann_ = str(path_ann_, "utf-8")

        full_path = join(self.root_path, path_ann_)
        ann = json.load(open(full_path, 'r'))

        # add filename to target:  # TODO: maybe instead ensure all info is in the JSON...
        ann['filename'] = path_ann_.split('/')[-1]

        return ann


class YTBBDiskDataset(DiskDataset):

    def __init__(self,
                 root_path='/mnt/disks/datasets',
                 paths_dataframe=None,
                 transform=None,
                 target_transform=None):
        super().__init__()
        self.root_path = root_path
        self.df_paths = paths_dataframe
        self.transform = transform
        self.target_transform = target_transform
        self.youtube_ids = np.unique(self.df_paths.youtube_id.values)

    def __getitem__(self, index):

        # This block takes <10ms:
        sampled_id = np.random.choice(self.youtube_ids)
        df_sampled = self.df_paths[self.df_paths.youtube_id == sampled_id]  # only frames from `youtube_id` clip
        sampled_frames = df_sampled.sample(n=2, replace=False).values.astype(str)  # numpy
        sampled_paths = [join(prefix, id_ + '.' + postfix) for prefix, id_, postfix in sampled_frames]

        imgs = list()
        tgts = list()
        for pth in sampled_paths:
            img = self.get_img(pth)

            # get annotation path from image path:
            target_path = pth.replace('/images/', '/annotations/').replace('.jpg', '.json')
            target = self.get_target(target_path)

            if img is None or target is None:
                return [None], [None]  # because if 2 examples, gets completely screwed up...

            if self.transform is not None:
                img = self.transform(image=img)['image']

            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.joint_transform is not None:
                img, target = self.joint_transform(img, target)

            imgs.append(img)
            tgts.append(target)

        return imgs, tgts

    def __len__(self):
        return len(self.youtube_ids)


class GCSDataset(Dataset):
    """Generic dataset for GCS.

    """

    def __init__(self,
                 bucketname='imidatasets',
                 paths_image=None,
                 paths_annotation=None,
                 target_list=None,
                 transform=None,
                 target_transform=None,
                 joint_transform=None,
                 cache_path=None,
                 debug_mode=False,
                 use_torchvision=False):  # TODO: stupid hack
        """Custom dataset for GCS.

        :param paths_image: list or numpy array of `path`s where path is of form 'folder/.../image_filename.jpg'
        :param paths_annotation: list or numpy array of `path`s to a json file, where path is of form
            'folder/.../annotation_filename.json'
        :param target_list: list, numpy array or dict of targets corresponding to each path in `path_list`; ndim >= 1 as
            for `path_list`
        :param transform:
        :param target_transform:
        :param joint_transform: joint transform for both image and target containing e.g. keypoints, segmaps, ...
        :param cache_data: whether or not cache data locally to enable faster read access once entire dataset has
                been cached.
        :param
        """

        assert not (paths_annotation is not None and target_list is not None), \
            'Must provide either annotation file paths or explicit targets!'

        self.paths_image = paths_image
        self.target_list = target_list
        self.paths_annotation = paths_annotation
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.cache_path = cache_path
        self.debug_mode = debug_mode
        self.use_torchvision = use_torchvision
        if debug_mode:
            print(colored('WARNING: RUNNING DATALOADER IN DEBUG MODE!!', 'red'))

        os.makedirs(self.cache_path, exist_ok=True) if self.cache_path is not None else None

        self.bucketname = bucketname
        self.store = None
        self.bucket = None

    def start_bundle(self):
        """This is required for multiprocess distributed training since multiprocessing can't pickle
        storage.Client() objects, see here:
        https://github.com/googleapis/google-cloud-python/issues/3191
        Also here: https://stackoverflow.com/a/59043240/742616

        The method will be run the first time __getitem__ is called.

        :return:
        """
        self.store = storage.Client()
        self.bucket = self.store.bucket(self.bucketname)

    def __len__(self):
        return len(self.paths_image)

    def __getitem__(self, index):

        if self.store is None:  # instantiate storage clients
            self.start_bundle()

        blob_path = self.paths_image[index]  # e.g. 'imagenet/train/xxxx.jpg'
        img = self.get_img(blob_path)

        if self.target_list is not None:
            target = self.target_list[index]
        elif self.paths_annotation is not None:
            target_path = self.paths_annotation[index]
            target = self.get_target(target_path)
        else:
            target = None

        if img is None:
            return None, None

        if self.transform is not None:
            if self.use_torchvision:
                img = self.transform(img)
            else:
                img = self.transform(image=img)['image']

        if self.target_transform is not None:
            target = self.target_transform(target)  # TODO: possibly handle None

        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        return img, target

    def get_img(self, blob_path_, convert_to_rgb=True):
        """Load an image from GCS.

        :param blob_path_:
        :param convert_to_rgb:
        :return:
        """

        if self.debug_mode:  # just generate a random "image" for debugging device speed
            img = np.random.randn(224, 224, 3).astype(np.uint8)
            img = Image.fromarray(img).convert("RGB")
            return np.array(img)

        if isinstance(blob_path_, bytes):
            blob_path_ = str(blob_path_, "utf-8")

        local_path, filename = split_by_at(blob_path_, "/", -1)
        ext = filename.split(".")[-1].lower()

        if self.cache_path is not None:
            cache_folder = join(self.cache_path, local_path)
            cache_file_path = join(cache_folder, filename)
            os.makedirs(cache_folder, exist_ok=True)
            cached_file_is_present = os.path.exists(cache_file_path)
        else:
            cache_file_path = None
            cached_file_is_present = False

        if not cached_file_is_present:  # from blob storage if no local fname_img_
            try:  # TODO: use the lower level functions
                blob = self.bucket.blob(blob_path_)
                assert blob.exists(), 'Blob does not exist!'
                img_bytes = blob.download_as_string()
                stream = io.BytesIO(img_bytes)
            except Exception as e:  # TODO: handle exceptions
                print(
                    "blob_path_:",
                    blob_path_,
                    "local_path:",
                    local_path,
                    "filename:",
                    filename,
                )
                print("**** MISSING BLOB:   " + blob_path_)
                print(e)
                return None  # collate_fn will handle missing examples
        else:
            stream = None

        try:
            if ext in ["jpeg", "jpg", "png", "gif"]:  # TODO: not cool
                if stream is not None:
                    img = Image.open(stream)
                else:
                    img = Image.open(cache_file_path)  # TODO: ughh not elegant...
                img = img.convert("RGB") if convert_to_rgb else img.convert("I")  # ensure RGB *BEFORE* saving
                if self.cache_path is not None and stream is not None:  # save if using cache
                    img.save(cache_file_path)

            else:
                print("**** CANNOT HANDLE FILETYPE:   " + blob_path_)
                stream.close() if stream is not None else None
                return None  # collate_fn will handle missing examples

            stream.close() if stream is not None else None

        except OSError as err:
            print(f"Exception while attempting to read image: {err}...", end='')
            try:
                os.remove(cache_file_path)
                print(' deleted!')
            except Exception as err2:
                print(f' could not remove: {err2}')
            stream.close() if stream is not None else None
            return None  # collate_fn will handle missing examples

        if self.use_torchvision:
            return img
        else:
            return np.array(img)  # TODO: somehow not converted to numpy for *streamed* png files...

    def get_target(self, blob_path_):
        """Load a JSON annotation file from GCS.
        # TODO: support also segmentation maps etc.
        :param blob_path_:
        :return:
        """

        if isinstance(blob_path_, bytes):
            blob_path_ = str(blob_path_, "utf-8")

        local_path, filename = split_by_at(blob_path_, "/", -1)
        if self.cache_path is not None:
            cache_folder = join(self.cache_path, local_path)
            cache_file_path = join(cache_folder, filename)
            os.makedirs(cache_folder, exist_ok=True) if self.cache_path is not None else None
            cached_file_is_present = os.path.exists(cache_file_path)
        else:
            cache_file_path = None
            cached_file_is_present = False

        ext = filename.split(".")[-1].lower()

        if not cached_file_is_present:  # from blob storage if no local file
            try:
                blob = self.bucket.blob(blob_path_)
                assert blob.exists(), 'Blob does not exist!'
                ann = eval(str(blob.download_as_string(), encoding='utf-8'))
            except Exception as e:  # TODO: handle exceptions
                print(
                    "cache_file_path:",
                    cache_file_path,
                    "blob_path_:",
                    blob_path_,
                    "local_path:",
                    local_path,
                    "filename:",
                    filename,
                )
                print("**** MISSING BLOB:   " + blob_path_)
                print(e)
                return None  # collate_fn will handle missing examples
            # Save locally:
            if self.cache_path is not None:
                json.dump(ann, open(cache_file_path, 'w'))
        else:
            ann = json.load(open(cache_file_path, 'r'))

        # add filename to target:  # TODO: maybe edit jsons instead...
        ann['filename'] = filename

        return ann


class RandomDataset(Dataset):

    def __init__(self, args):
        self.size = args.input_size
        self.batch_size = args.batch_size
        self.dataset_len = args.dataset_len

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        img = torch.rand(self.batch_size, 3, self.size, self.size, dtype=torch.float32)
        tgt = torch.ones(self.batch_size)
        tgt = dict(class_id=tgt)
        return img, tgt


class YoutubeDataset(GCSDataset):

    def __init__(self,
                 bucketname='imidatasets',
                 paths_dataframe=None,
                 transform=None,
                 target_transform=None,
                 intraclip_examples=1,
                 cache_path=None,
                 debug_mode=False,
                 image_folder=None):
        """Youtube-bb dataset multi-indexed by (`youtube_id`, `timestamp`). Dataset length is the number of unique
        `youtube_id`s. After sampling N `youtube_id`s, __getitem__() will sample `intraclip_examples` number of frames
        from within each clip.

        :param bucketname:
        :param paths_dataframe: dataframe with columns `prefix, youtube_id, postfix`
        :param transform:
        :param target_transform:
        :param intraclip_examples: number of examples to draw from within one clip
        :param image_folder: maybe override default 'images' folder and instead use e.g. the low resolution
            'images_128x' folder
        """
        super().__init__()

        self.bucketname = bucketname
        self.df_paths = paths_dataframe
        self.transform = transform
        self.target_transform = target_transform
        self.intraclip_examples = intraclip_examples
        self.youtube_ids = np.unique(self.df_paths.youtube_id.values)
        self.cache_path = cache_path
        self.debug_mode = debug_mode
        self.image_folder = image_folder

    def __getitem__(self, index):

        if self.store is None:  # instantiate storage clients
            self.start_bundle()

        # This block takes <10ms:
        sampled_id = np.random.choice(self.youtube_ids)
        df_sampled = self.df_paths[self.df_paths.youtube_id == sampled_id]  # only frames from `youtube_id` clip
        num_examples = min(self.intraclip_examples, len(df_sampled))
        sampled_frames = df_sampled.sample(n=num_examples, replace=False).values.astype(str)  # numpy
        sampled_paths = [join(prefix, id_ + '.' + postfix) for prefix, id_, postfix in sampled_frames]

        imgs = list()
        tgts = list()
        for pth in sampled_paths:

            if self.image_folder is not None:  # use alternative resolution images
                pth_img = pth.replace('/images/', '/' + self.image_folder + '/')
            else:
                pth_img = pth
            img = self.get_img(pth_img)

            # get annotation path from image path:
            target_path = pth.replace('/images/', '/annotations/').replace('.jpg', '.json')
            target = self.get_target(target_path)

            if img is None or target is None:
                return [None], [None]  # because if 2 examples, gets completely screwed up...

            if self.transform is not None:
                img = self.transform(image=img)['image']

            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.joint_transform is not None:
                img, target = self.joint_transform(img, target)

            imgs.append(img)
            tgts.append(target)

        return imgs, tgts

    def __len__(self):
        return len(self.youtube_ids)


class InferenceDataset(GCSDataset):

    def __init__(self,
                 bucketname='imidatasets',
                 paths_image=None,
                 transform=None,
                 cache_path=None,
                 # sizes=[256],
                 pad_to_mult=32,
                 max_size=512):
        """
        NOTE: make sure transforms are same as during training!

        :param bucketname:
        :param paths_image:
        :param sizes: list of ints = square shapes at which to do the multi-scale inference
        """
        super().__init__()
        self.cache_path = cache_path
        self.bucket_name = bucketname
        self.transform = transform
        # self.sizes = sizes
        self.path_list = paths_image
        self.pad_to_mult = pad_to_mult
        self.max_size = max_size

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):
        if self.store is None:  # instantiate storage clients
            self.start_bundle()

        blob_path = self.path_list[index]

        img = self.get_img(blob_path)  # numpy array, shape [H, W, C]

        # Apply base (albumentations) augs:
        img = self.transform(image=img)['image']  # TODO: this is very slow...

        original_shape = img.shape[:2]

        if img is None:
            return None, None, None

        # Resize longest side to max size:
        # if np.any(np.array(original_shape) > self.max_size):
        long_size = np.max(original_shape)
        short_side = np.min(original_shape)
        new_size = (self.max_size, int((self.max_size / long_size) * short_side))
        new_size = (new_size[1], new_size[0]) if original_shape[0] > original_shape[1] else new_size
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)  # note: WxH not HxW!?!?

        # Pad to square size:
        img_padded = np.zeros([self.max_size, self.max_size, 3], dtype=np.float32)
        img_padded[:new_size[1], :new_size[0]] = img

        return img_padded, blob_path, original_shape


class OpenImagesDataset(GCSDataset):

    def __init__(self,
                 bucketname='imidatasets',
                 paths='metadata_train.npz',
                 transform=None,
                 cache_path=None,
                 debug_mode=False):
        """It's assumed that the images, masks are in subfolders `train`, `train_masks` etc.
        For now supports only segmentation masks.

        :param bucketname:
        :param paths: array of byte strings corresponding to the image path, `/{bucket}/{folder}/{image_id}.jpg`
        :param transform: imgaug transforms
        :param cache_path:
        :param debug_mode:
        """
        super().__init__()
        self.bucketname = bucketname
        self.paths = paths
        self.transform = transform
        self.cache_path = cache_path
        self.debug_mode = debug_mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        if self.store is None:  # instantiate storage clients
            self.start_bundle()

        path_img = str(self.paths[index], encoding='utf-8')  # `/{bucket}/{folder}/{image_id}.jpg`
        path_mask = self.get_mask_path(path_img)  # `/{bucket}/{folder}_masks/{image_id}_mask.png`

        # Get image:
        img = self.get_img(path_img)

        # Get mask:
        mask = self.get_img(path_mask, convert_to_rgb=False)

        if (img is None) or (mask is None):
            return None, None  # collate function will handle Nones

        if self.transform is not None:
            img, mask = self.transform(image=img, mask=mask)

        return img, mask

    def get_mask_path(self, path_img_):
        path_mask = path_img_.split('.')[0] + '_mask.png'  # `/{bucket}/{folder}/{image_id}_mask.png`
        path_split = path_mask.split('/')
        path_split[1] = path_split[1] + '_masks'
        path_mask = '/'.join(path_split)  # `/{bucket}/{folder}_masks/{image_id}_mask.png`
        return path_mask




def extract_imagenet_classes(paths):
    """Will extract the Wordnet ids (wnids) from the paths, e.g.
    'n01440764' from 'train/n01440764/n01440764_10026.JPEG' and so on.

    :param paths: list or numpy array of paths.
    :return:
    """

    wnids = []
    for path in paths:
        wnid = path.split("/")[1]
        wnids.append(wnid)
    wnids = np.array(wnids)

    classes = np.sort(np.unique(wnids))
    num_classes = len(classes)

    wnid_to_idx = {classes[i]: i for i in range(num_classes)}

    target_list = np.array([wnid_to_idx[wnid] for wnid in wnids])

    return target_list


def get_gcs_metadata(args):
    metadata_val = load_numpy_from_gcs(args.bucket_name, args.metadata_val_filename)
    metadata_trn = load_numpy_from_gcs(args.bucket_name, args.metadata_train_filename)

    paths_val = metadata_val['paths']
    targets_val = metadata_val['targets']
    paths_trn = metadata_trn['paths']
    targets_trn = metadata_trn['targets']

    metadata_val.close()
    metadata_trn.close()

    return paths_trn, targets_trn, paths_val, targets_val


def get_openimages_metadata(args):
    paths_trn = load_numpy_from_gcs(args.bucket_name, args.metadata_train_filename)['paths']
    paths_val = load_numpy_from_gcs(args.bucket_name, args.metadata_val_filename)['paths']

    return paths_trn, paths_val


def get_deepfashion_metadata(args):
    pair_ids_trn = load_numpy_from_gcs(args.bucket_name, args.metadata_train_filename)
    pair_ids_val = load_numpy_from_gcs(args.bucket_name, args.metadata_val_filename)

    return pair_ids_trn, pair_ids_val


"""Stuff below forked from torchvision and modified by harpone
TODO maybe insert license

"""


def has_file_allowed_extension(filename, extensions):
    """Checks if a fname_img_ is an allowed extension.

    Args:
        filename (string): path to a fname_img_

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    if len(classes) > 0:
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
    else:
        classes = ["None"]
        class_to_idx = {"None": 0}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    if len(class_to_idx) > 1:
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
    else:
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, 0)
                    images.append(item)

    return images


class DatasetFolder(Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transform.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transform it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]


def pil_loader(path):
    # open path as fname_img_ to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class WNID2Idx(object):
    def __init__(self, wnids):
        self.wnid_to_idx = {wnids[i]: i for i in range(len(wnids))}

    def __call__(self, wnid):
        return np.int32(self.wnid_to_idx[wnid])


def get_imagenet_dataloader(args, phase='train'):
    '''

    :param args:
    :param phase:
    :param world_size: number of devices
    :return:
    '''
    min_crop_scale = args.get('min_crop_scale', 0.08)
    if phase == 'train':
        data_path = args.metadata_train_filename
        transform = alb.Compose(
            [  # alb.IAACropAndPad(percent=(0, 0.15), keep_size=False),  # so sees some padding also during training
                alb.RandomResizedCrop(args.input_size,
                                      args.input_size,
                                      scale=(min_crop_scale, 1.),  # > 1 scales don't work apparently...
                                      ratio=(0.7, 1.4),
                                      always_apply=True,
                                      interpolation=cv2.INTER_NEAREST),
                # alb.Rotate(limit=45, p=0.5, border_mode=0, interpolation=cv2.INTER_NEAREST),
                alb.HorizontalFlip(p=.5),
                # alb.Equalize(always_apply=True),
                alb.Normalize(mean=0, std=1, always_apply=True),  # to unit interval
                ToTensorV2()],
            p=1
        )

    elif phase == 'validate':
        data_path = args.metadata_val_filename
        # TODO: not quite ImageNet-resnet validation... first to 256, then center crop...
        transform = alb.Compose([alb.SmallestMaxSize(args.input_size, interpolation=cv2.INTER_NEAREST),
                                 alb.CenterCrop(args.input_size, args.input_size, always_apply=True),
                                 # alb.Equalize(always_apply=True),
                                 alb.Normalize(mean=0, std=1, always_apply=True),  # to unit interval
                                 ToTensorV2()],
                                p=1)

    else:
        raise ValueError

    dataset = LMDBDataset(data_path, transform=transform, target_transform=None)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=phase == 'train',
        sampler=None,
        num_workers=args.num_workers,
        drop_last=phase == 'validate',
        collate_fn=collate_general,
        pin_memory=False,
    )

    return dataloader


def get_openimages_dataloader(args, phase='train', world_size=1, rank=0):
    if phase == 'train':
        if rank == 0:
            print('Getting training dataloader.')  # TODO: border mode = zero padding!!
        # WARNING: need to pay special attention to interpolation to avoid wrong classes on mask boundaries!
        # Also `keep_size=False` is important in IAACropAndPad
        transform = alb.Compose(
            [alb.IAACropAndPad(percent=(0, 0.15), keep_size=False),  # so sees some padding also during training
             alb.RandomResizedCrop(args.input_size,
                                   args.input_size,
                                   scale=(0.5, 1.25),  # > 1 scales don't work apparently...
                                   ratio=(0.7, 1.4),
                                   always_apply=True,
                                   interpolation=cv2.INTER_NEAREST),
             alb.Rotate(limit=45, p=0.5, border_mode=0, interpolation=cv2.INTER_NEAREST),
             alb.HorizontalFlip(p=.5),
             alb.Equalize(always_apply=True),
             alb.Normalize(always_apply=True),
             # alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
             #                         brightness_by_max=True, always_apply=False, p=0.5),
             # alb.Posterize(num_bits=4, always_apply=False, p=0.5),

             # alb.MotionBlur(blur_limit=7, always_apply=False, p=0.5),
             # alb.HueSaturationValue(),
             # alb.Blur()
             ],
            p=1
        )
        paths, _ = get_openimages_metadata(args)
        # Limit training set size:
        paths = paths[:args.train_size]
    else:
        if rank == 0:
            print('Getting validation dataloader.')
        transform = alb.Compose([alb.SmallestMaxSize(args.input_size, interpolation=cv2.INTER_NEAREST),
                                 alb.CenterCrop(args.input_size, args.input_size, always_apply=True),
                                 alb.Equalize(always_apply=True),
                                 alb.Normalize(always_apply=True),
                                 ],
                                p=1)
        _, paths = get_openimages_metadata(args)
        # Limit validation set size:
        paths = paths[:args.eval_size]

    # Compute output sizes:
    transform_joint = OpenImagesTransform(transform, fmap_scales=args.fmap_scales)

    dataset = OpenImagesDataset(bucketname='imidatasets',
                                paths=paths,
                                transform=transform_joint,
                                cache_path=args.cache_path,
                                debug_mode=False)

    collate_fn = collate_with_masks

    # if args.process_count_per_node > 1:
    if world_size > 1:
        # print(colored('Multi-processing DistributedSampler', 'green'))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    return dataset, loader, sampler


def get_deepfashion_dataloader(args, phase='train', world_size=1, rank=0):
    if phase == 'train':
        if rank == 0:
            print('Getting training dataloader.')  # TODO: border mode = zero padding!!
        transform = alb.Compose([  # alb.Equalize(always_apply=True),
            alb.IAACropAndPad(percent=(0, 0.1)),
            alb.RandomResizedCrop(args.input_size,
                                  args.input_size,
                                  scale=(0.5, 1.25),  # > 1 scales don't work apparently...
                                  ratio=(0.7, 1.4),
                                  always_apply=True),
            # alb.Rotate(limit=45, p=0.75, border_mode=0),
            alb.HorizontalFlip(p=.5),
            alb.Normalize(always_apply=True),  # HRNet uses default params
            # alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
            #                         brightness_by_max=True, always_apply=False, p=0.5),
            # alb.Posterize(num_bits=4, always_apply=False, p=0.5),

            # alb.MotionBlur(blur_limit=7, always_apply=False, p=0.5),
            # alb.HueSaturationValue(),
            # alb.Blur()
        ],
            p=1,
            keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False)
        )
        pair_ids, _ = get_deepfashion_metadata(args)

        # Limit training set size:
        pair_ids = pair_ids[:args.database_size_train]
        root_folder = 'train'
    else:
        if rank == 0:
            print('Getting validation dataloader.')
        transform = alb.Compose([  # alb.Equalize(always_apply=True),
            alb.SmallestMaxSize(args.input_size),
            alb.CenterCrop(args.input_size, args.input_size, always_apply=True),
            alb.Normalize(always_apply=True)
        ],
            p=1,
            keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False))
        _, pair_ids = get_deepfashion_metadata(args)

        # Limit validation set size:
        pair_ids = pair_ids[:args.query_size_train]
        root_folder = 'validation'

    # Compute output sizes:
    joint_transform = DeepfashionTransform(transform, fmap_scales=args.fmap_scales)

    dataset = DeepfashionDataset(bucketname=args.bucket_name,
                                 pair_ids=pair_ids,
                                 root_folder=root_folder,
                                 joint_transform=joint_transform,
                                 # cache_path=args.cache_path,
                                 debug_mode=False)

    collate_fn = collate_deepfashion

    # if args.process_count_per_node > 1:
    if world_size > 1:
        # print(colored('Multi-processing DistributedSampler', 'green'))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    return dataset, loader, sampler


def get_dataloader(args, phase='train'):
    if args.dataset is 'youtube':
        loader = get_youtube_dataloader(args, phase=phase)
    elif args.dataset is 'youtube_wds':
        loader = get_youtube_wds_dataloader(args, phase=phase)
    elif 'CIFAR10' in args.dataset:
        loader = get_cifar10_dataloader(args, phase=phase)
    elif 'imagenet' in args.dataset:
        loader = get_imagenet_dataloader(args, phase=phase)
    elif 'openimages' in args.dataset:
        loader = get_openimages_wds_dataloader(args, phase=phase)
    else:
        raise NotImplementedError

    return loader


def get_cifar10_dataloader(args, phase='train'):  # TODO: WiP
    from torchvision.datasets import CIFAR10
    if phase == 'train':
        transform = alb.Compose([alb.RandomResizedCrop(32,
                                                       32,
                                                       scale=(0.5, 1),
                                                       ratio=(3 / 4, 4 / 3),
                                                       always_apply=True),
                                 alb.HorizontalFlip(p=.5),
                                 alb.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.8,
                                                              brightness_by_max=True, p=0.8),
                                 alb.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                                        p=0.8),
                                 alb.ToGray(p=0.2),
                                 alb.GaussianBlur(blur_limit=7, p=0.5),
                                 alb.Normalize(mean=0, std=1, always_apply=True),  # to unit interval
                                 ],
                                p=1,
                                # keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False)
                                )
    elif phase == 'validate':
        transform = alb.Compose([  # alb.Equalize(always_apply=True),
            alb.SmallestMaxSize(32),
            alb.CenterCrop(32, 32, always_apply=True),
            alb.Normalize(mean=0, std=1, always_apply=True),  # to unit interval
        ],
            p=1,
            # keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False)
        )
    else:
        raise ValueError

    dataset = CIFAR10('.', train=phase == 'train', transform=transform, download=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=phase == 'train',
        sampler=None,
        num_workers=args.num_workers,
        drop_last=phase != 'train',
        collate_fn=collate_ytbb,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return loader


def get_ytbb_id(key_):
    """Get clip id from `key_`, which is the filename without extension.

    :param key_: e.g. 'images/v_xasjbaaolsn_mp4_28000'
    :return: key: str ('xasjbaaolsn'), dt: int (28000, timestamp in ms)
    """
    parts = key_.split('/')[1]
    key = parts[2:13]
    dt = parts[18:]
    return key, dt


def filter_nones(src, has_keys=None):
    """Some images, targets may be missing or None because of corrupt data, so filter them out.

    For webdatasets only.

    :param src: generator outputting a dict with keys `__key__`, `jpg`, `json`
    :return:
    """
    # TODO: maybe keys as kwarg
    for sample in src:
        no_nones = not any([value is None for _, value in sample.items()])
        if has_keys:
            has_all_keys = all(val in sample.keys() for val in has_keys)
        else:
            has_all_keys = True
        if has_all_keys and no_nones:
            yield sample


def get_youtube_wds_dataloader(args, phase='train'):
    if phase == 'train':
        transform = alb.Compose([alb.RandomResizedCrop(args.input_size,
                                                       args.input_size,
                                                       scale=(0.08, 1),
                                                       ratio=(3 / 4, 4 / 3),
                                                       always_apply=True),
                                 # alb.Rotate(limit=45, p=0.75, border_mode=0),
                                 alb.HorizontalFlip(p=.5),
                                 alb.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.8,
                                                              brightness_by_max=True, p=0.5),
                                 alb.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                                        p=0.8),
                                 alb.ToGray(p=0.25),
                                 alb.Normalize(mean=0, std=1, always_apply=True),  # to unit interval
                                 ToTensorV2()  # TODO: need this?
                                 ],
                                p=1,
                                # keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False)
                                )
        urls = 'gs://imidatasets/youtube-bb-full/tars/ytbb-{000000..000240}.tar'

    elif phase == 'validate':
        transform = alb.Compose([  # alb.Equalize(always_apply=True),
            alb.SmallestMaxSize(args.input_size),
            alb.CenterCrop(args.input_size, args.input_size, always_apply=True),
            alb.Normalize(mean=0, std=1, always_apply=True),  # to unit interval
            ToTensorV2()
        ],
            p=1,
            # keypoint_params=alb.KeypointParams(format='xy', remove_invisible=False)
        )
        urls = 'gs://imidatasets/youtube-bb-full/tars/ytbb-{000240..000247}.tar'

    def transform_fn(img):
        return transform(image=img)['image']

    def identity_fn(x):
        return x

    def generate_pairs(src, sample_fraction=args.pairs_sample_fraction):
        """Generate example pairs by sampling from within same clip.

        For webdatasets only.

        NOTE: None:s supposed to be removerd by `filter_nones`, so can use the default webdataset collation function!

        :param src: generator yielding (list(keys), list(inputs), list(targets))
        :param sample_fraction: fraction of sample pairs returned prop to all possible pairs. Low sample_fraction => more
            diverse minibatches, but slower.
        :return:
        """
        for keys, inputs, targets in src:
            ids_timestamps = np.array([get_ytbb_id(key) for key in keys])
            ids_len = len(ids_timestamps)
            ids_unique = np.unique(ids_timestamps[:, 0])
            ids = ids_timestamps[:, 0]
            timestamps = ids_timestamps[:, 1]
            for id_ in ids_unique:
                idx_this = np.arange(ids_len)[ids == id_]  # e.g. [0, 1, 2]
                num_samples = len(idx_this)
                num_pairs = int(np.round(sample_fraction * num_samples * (num_samples + 1) / 2))  # incl. same idx
                for _ in range(num_pairs):
                    frame1, frame2 = np.random.choice(idx_this, size=2, replace=True)  # can sample same indices now
                    # Add id for sanity checks:
                    out_1 = targets[frame1]
                    out_1['id'] = id_
                    out_1['timestamp'] = timestamps[frame1]
                    out_2 = targets[frame2]
                    out_2['id'] = id_
                    out_2['timestamp'] = timestamps[frame2]
                    yield inputs[frame1], out_1, inputs[frame2], out_2

    if 1:  # use gsutil TODO warning getting ValueError: invalid literal for int() with base 10: '1031684608, 414998528' sometimes
        urls = f'pipe:gsutil cp {urls} -'  # or 'gsutil cat {urls}' but cp has some checksum stuff I think
    if 0:  # curl with authentication
        urls = replace_gcs_endpoint(urls)  # still in brace form
        urls = f'pipe:curl -L -s -H "Authorization: Bearer "$(gcloud auth application-default print-access-token) {urls} || true'

    if args.use_random_data:  # feed in random data for debugging
        dataset = RandomDataset(args)
    else:
        # tarhandler=warn_and_continue because sometimes rarely corrupt jpg
        shuffle_buffer = args.shuffle_buffer if phase == 'train' else 10
        dataset = (wds.Dataset(urls, length=None, tarhandler=warn_and_continue)
                   .decode('rgb8')  # dict with keys: __key__, jpeg, json
                   .pipe(filter_nones)
                   .to_tuple('__key__', 'jpg', 'json', handler=wds.warn_and_continue)
                   .map_tuple(identity_fn, transform_fn, identity_fn)
                   .batched(args.pairs_batch_size)
                   .pipe(generate_pairs)
                   # .shuffle(args.shuffle_buffer, rng=NumpyRNG())
                   .pipe(shuffle(shuffle_buffer, initial=shuffle_buffer, rng=utils.NumpyRNG()))
                   .pipe(batched(batchsize=args.batch_size, partial=True, collation_fn=collate_ytbb_wds))
                   )
        # TODO: debugging!!!!
        # nominal = args.nominal_dataset_len // args.batch_size // (args.tpu_cores or args.gpus)
        # dataset = wds.ResizedDataset(dataset, length=5000000, nominal=nominal)  # no need for ResizedDataset if Multidataset

    if 0:  # basic DataLoader
        def collate_id(x_): return x_

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # because batching done in dataset
            shuffle=False,
            num_workers=args.num_workers if phase == 'train' else 1,  # 1 for val because OOM easily
            drop_last=True,
            collate_fn=collate_id,
            pin_memory=args.tpu_cores is None,
            worker_init_fn=worker_init_fn,
        )
    if 1:

        def unbatch_dct(data):
            """
            TODO: shit not all tensors in dict have batch dim :/
            :param data: MultiDatasetIterator with (x_batch, y_batch=dict)
            :yield: one example pair (x, y)
            """
            for sample in data:
                assert isinstance(sample, (tuple, list)), sample
                assert len(sample) > 0
                for i in range(len(sample[0])):
                    yield sample[0][i], {key: val[i] for key, val in sample[1].items()}

        loader = (wds.MultiDataset(dataset,
                                   workers=args.num_workers if phase == 'train' else 1,  # 1 for val because OOM easily
                                   pin_memory=False,  # problems with dict targets
                                   output_size=10000)
                  .pipe(unbatch_dct)
                  .pipe(shuffle(shuffle_buffer, initial=shuffle_buffer, rng=NumpyRNG()))
                  .pipe(batched(batchsize=args.batch_size, partial=True, collation_fn=recollate_ytbb))
                  )

    return loader


def transform_openimages(src, aug=None):
    """Apply Albumentations transformations to `image`, `mask` and bboxes in `target`.
    Bounding boxes are also transformed to numpy array vectors.

    scratchpad for debugging:
    num_ids_present = np.arange(len(mask_bbox))[mask_bbox.sum(-1).sum(-1) > 0]

    :param src:
    :return:
    """
    for image, mask, target in src:
        if aug:
            # Gather bboxes as list of [x_min, y_min, x_max, y_max], relative coords:
            x_mins = target.get('XMin', [])
            y_mins = target.get('YMin', [])
            x_maxs = target.get('XMax', [])
            y_maxs = target.get('YMax', [])
            x_min1s = target.get('XMin1', [])
            y_min1s = target.get('YMin1', [])
            x_max1s = target.get('XMax1', [])
            y_max1s = target.get('YMax1', [])
            x_min2s = target.get('XMin2', [])
            y_min2s = target.get('YMin2', [])
            x_max2s = target.get('XMax2', [])
            y_max2s = target.get('YMax2', [])
            bboxes = list(zip(x_mins, y_mins, x_maxs, y_maxs, ['bbox'] * len(x_mins)))
            bboxes += list(zip(x_min1s, y_min1s, x_max1s, y_max1s, ['bbox1'] * len(x_min1s)))
            bboxes += list(zip(x_min2s, y_min2s, x_max2s, y_max2s, ['bbox2'] * len(x_min2s)))

            # Need to include bbox labels in aug since bboxes may be dropped:
            labels_bbox = target['LabelNameBB']
            labels_1 = target['LabelName1']
            labels_2 = target['LabelName2']
            bbox_labels = labels_bbox + labels_1 + labels_2

            augmented = aug(image=image, mask=mask, bboxes=bboxes, bbox_labels=bbox_labels)
            image = augmented['image']
            mask = augmented['mask']
            bboxes = augmented['bboxes']
            bbox_labels = augmented['bbox_labels']
            bboxes_base = np.array([bbox[:4] for bbox in bboxes if bbox[-1] == 'bbox'])
            bboxes_1 = np.array([bbox[:4] for bbox in bboxes if bbox[-1] == 'bbox1'])
            bboxes_2 = np.array([bbox[:4] for bbox in bboxes if bbox[-1] == 'bbox2'])

            # Process bounding box coords:
            if len(bboxes_base) > 0:
                target['XMin'] = bboxes_base[:, 0]
                target['YMin'] = bboxes_base[:, 1]
                target['XMax'] = bboxes_base[:, 2]
                target['YMax'] = bboxes_base[:, 3]
            else:
                target['XMin'] = np.array([0.])
                target['YMin'] = np.array([0.])
                target['XMax'] = np.array([0.])
                target['YMax'] = np.array([0.])

            # Maybe process visual relation bbox coords:
            if len(bboxes_1) > 0 and len(bboxes_2) > 0:
                target['XMin1'] = bboxes_1[:, 0]
                target['YMin1'] = bboxes_1[:, 1]
                target['XMax1'] = bboxes_1[:, 2]
                target['YMax1'] = bboxes_1[:, 3]
                target['XMin2'] = bboxes_2[:, 0]
                target['YMin2'] = bboxes_2[:, 1]
                target['XMax2'] = bboxes_2[:, 2]
                target['YMax2'] = bboxes_2[:, 3]

            # Get label indices for xent loss:
            labels_img = np.array(target['LabelNameImage'])
            labels_img_int = np.array([name2idx_bbox[lbl] for lbl in labels_img])
            label_presence = np.array(target['LabelPresence'])
            positive_labels_int = labels_img_int[label_presence == 1]
            negative_labels_int = labels_img_int[label_presence == 0]
            negative_labels = labels_img[label_presence == 0]  # will be used in centerness masks also
            labels_img_vec = np.zeros([len(idx2name_bbox), ], dtype=np.float32)
            labels_img_vec.fill(np.nan)  # NaN means not present
            labels_img_vec[positive_labels_int] = 1
            labels_img_vec[negative_labels_int] = -1

            target['LabelIntImage'] = labels_img_int
            target['LabelVec'] = labels_img_vec

            # Get label indices for bboxes:
            labels_bb_int = np.array([name2idx_bbox[lbl] for lbl in bbox_labels[:len(bboxes_base)]])
            target['LabelIntBB'] = labels_bb_int

            # Get label indices for relations:
            # TODO oops these are the relation tags... finish up later if needed
            # TODO note need bbox_labels[len(bboxes_base):len(bboxes_base) + len(bboxes_1)] etc
            # labels_1_int = np.array([name2idx[lbl] for lbl in labels_1])
            # labels_2_int = np.array([name2idx[lbl] for lbl in labels_2])
            # target['LabelInt1'] = labels_1_int
            # target['LabelInt2'] = labels_2_int

            # Generate downsampled segmentation mask:
            mask = get_openimages_segmask(mask, negative_labels)

            # Generate downsampled bbox mask:
            mask_bbox = get_openimages_bbmask(bboxes_base, labels_bb_int, mask.shape[1:], negative_labels_int,
                                              num_classes_bbox)

            target['mask'] = mask  # [H, W], int valued
            target['mask_bbox'] = mask_bbox  # [num_classes, H, W], float valued

        yield image, target


def get_openimages_segmask(mask, negative_labels_):
    """Get positives/ negatives segmentation mask by using global negative labels.

    :param mask: original image shape int mask, shape [H, W]; values are the positive labels
    :param negative_labels_: list/array of ints
    :return:
    """
    mask = zoom(mask, 1 / 32, order=0)  # TODO: maybe 32 to args although prolly will be constant for all eternity...

    # Mask to pos/neg:
    mask_seg = np.zeros((num_classes_seg,) + mask.shape, dtype=np.float32)
    mask_seg.fill(np.nan)  # NaN = missing by default

    # Fill negative labels:
    neg_labels_int = [name2idx_seg.get(name, None) for name in negative_labels_]
    for neg_label in neg_labels_int:
        if neg_label is not None:
            mask_seg[neg_label] = -1

    # Fill positive labels:
    seg_labels = np.unique(mask)
    for seg_label in seg_labels:
        if seg_label > 0:
            mask_seg[seg_label][np.isnan(mask_seg[seg_label])] = -1  # -1 outside of segmentation
            mask_seg[seg_label][mask == seg_label] = 1

    return mask_seg


def get_openimages_bbmask(bboxes, labels, shape, negative_labels, num_classes):
    """Form a bbox centerness mask from a list/array of bboxes, accompanying labels and output mask shape.

    NOTE: I'm actually not generating an FCOS detection bbox regression targets to bbox edges but just
    a mask with distance to boundary per class. For each label and (feature-) pixel in [H, W], we have NaN if no
    positive or negative labels, -1 if there is an image level negative label, a value `centerness` > 0 if the pixel
    center is at distance `centerness` from the bbox boundary, or -1 if the feature pixel is outside of a bounding box.

    NOTE: think if it's better to actually discard values 0 outside bbox, since not all bboxes may be present?
    OpenImages does seem to have lots of bboxes though, so maybe I can keep them.

    NOTE 2: negative labels may not be needed, since I'm doing regression to centerness, which can't be guessed to be
    one as for binary xent.

    NOTE 3: or should I just take all nonexisting bboxes as negatives? FCOS did that! For rep learning, the former
    may be better though...

    :param bboxes: shape [num_bboxes, 4] where the last dim are the `XMin`, `YMin`, `XMax`, `YMax` coords.
    :param labels: int, shape [num_bboxes, ]
    :param shape: tuple (H, W)
    :param negative_labels: if not None or len() > 0, non-bbox pixels will be assigned these negative labels.
    :param num_classes:
    :return: shape [num_classes, H, W] centerness mask; values are distances to the nearest edge measured
        from center of a (feature) pixel.
    """
    # TODO: I could easily replace NaNs with -1 to force not present to be negatives!
    assert len(bboxes) == len(labels)

    height, width = shape
    h_centers = np.arange(height) + 0.5
    w_centers = np.arange(width) + 0.5
    result = np.zeros((num_classes,) + shape, dtype=np.float32)
    result.fill(np.nan)  # nan = does not contribute

    # add negative labels as -1:
    for neg_label in negative_labels:
        result[neg_label] = -1

    for label, bbox in zip(labels, bboxes):  # note: can be multiple same label bboxes!
        left = np.broadcast_to(w_centers[None] - bbox[0] * width, shape)  # [H, W]
        right = np.broadcast_to(bbox[2] * width - w_centers[None], shape)
        bottom = np.broadcast_to(bbox[3] * height - h_centers[:, None], shape)  # note bottom is last row!
        top = np.broadcast_to(h_centers[:, None] - bbox[1] * height, shape)
        distances2edges = np.stack([left, right, top, bottom], axis=0)  # [4, H, W]
        distances2edges[distances2edges < 0.] = 0  # discard distances outside bbox
        centerness = np.min(distances2edges, axis=0)  # [H, W]
        result[label][np.isnan(result[label])] = -1  # tag as negative by default
        result[label][centerness > 0.] = centerness[centerness > 0.]  # replace only positive parts
        # TODO: how about overlapping, same label bboxes? Pretty rare occurrence...

    return result


def collate_openimages(batch):
    """Packs `imgs` and `masks` to a minibatch tensor.
    :param batch: list of (image, target)
    :return:
    """
    # Unpack if lists:
    imgs = list()
    masks = list()
    masks_bbox = list()
    targets = collections.defaultdict(list)

    keys_used = ['LabelVec']  # has pos (+1), neg (-1) or not present (NaN) labels

    for image, target in batch:
        if image is None or target is None:  # maybe there was a read error/ corrupt example so skip
            continue
        imgs.append(torch.as_tensor(image, dtype=torch.float32))
        masks.append(torch.as_tensor(target['mask'], dtype=torch.float32))
        masks_bbox.append(torch.as_tensor(target['mask_bbox'], dtype=torch.float32))
        [targets[key].append(torch.as_tensor(target[key], dtype=torch.float32)) for key in keys_used]

    imgs = torch.stack(imgs, dim=0)
    masks = torch.stack(masks, dim=0)
    masks_bbox = torch.stack(masks_bbox, dim=0)
    targets['masks'] = masks
    targets['masks_bbox'] = masks_bbox

    for key in keys_used:
        targets[key] = torch.stack(targets[key], dim=0)

    return dict(images=imgs, targets=targets)


def decode_openimages(src):
    """Decode openimages data in the form of (image.jpg, mask.png, target.json) to
    (uint8 numpy array of shape [H, W, C], uint8 numpy array of shape [H, W], dict) respectively.

    Decode to uint8 numpy arrays because that's what albumentations works with.

    :param src:
    :return img, mask, target: img: uint8 ndarray shape [H, W, 3]; mask: int32 ndarray shape [H, W]; target: dict
    """

    for sample in src:
        try:
            img = sample['image.jpg']
            mask = sample['mask.png']
            target = sample['targets.json']
        except KeyError:  # not found
            continue
        with io.BytesIO(img) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = np.array(img.convert('RGB'))

        with io.BytesIO(mask) as stream:
            mask = PIL.Image.open(stream)
            mask.load()
            mask = np.array(mask)

        target = json.loads(target)

        # Filter nones now that all are loaded:
        if (img is None) or (mask is None) or (target is None):
            continue

        yield img, mask, target


def get_openimages_wds_dataloader(args, phase='train'):
    if phase == 'train':
        transform = alb.Compose([alb.RandomResizedCrop(args.input_size,
                                                       args.input_size,
                                                       scale=(0.2, 1),
                                                       ratio=(3 / 4, 4 / 3),
                                                       always_apply=True),
                                 # alb.Rotate(limit=45, p=0.75, border_mode=0),
                                 alb.MotionBlur(p=0.5),
                                 alb.HorizontalFlip(p=.5),
                                 alb.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.8, p=0.8),
                                 alb.ToGray(p=0.25),
                                 alb.Normalize(mean=0, std=1, always_apply=True),  # to unit interval
                                 ToTensorV2()  # TODO: need this?
                                 ],
                                p=1,
                                bbox_params=alb.BboxParams(format='albumentations')
                                )
        urls = 'gs://imidatasets/openimages-wds/train/openimages-c-train-{0..580}.tar'

    elif phase == 'validate':
        transform = alb.Compose([  # alb.Equalize(always_apply=True),
            alb.SmallestMaxSize(args.input_size),
            alb.CenterCrop(args.input_size, args.input_size, always_apply=True),
            alb.Normalize(mean=0, std=1, always_apply=True),  # to unit interval
            ToTensorV2()
        ],
            p=1,
            bbox_params=alb.BboxParams(format='albumentations', label_fields=['bbox_labels'])
        )
        urls = 'gs://imidatasets/openimages-wds/val/openimages-c-val-{0..9}.tar'

    if 1:  # use gsutil
        urls = f'pipe:gsutil cp {urls} -'  # or 'gsutil cat {urls}' but cp has some checksum stuff I think
    if 0:  # curl with authentication
        urls = replace_gcs_endpoint(urls)  # still in brace form
        urls = f'pipe:curl -L -s -H "Authorization: Bearer "$(gcloud auth application-default print-access-token) {urls} || true'

    if args.use_random_data:  # feed in random data for debugging
        dataset = RandomDataset(args)
    else:
        # tarhandler=warn_and_continue because sometimes rarely corrupt jpg
        def warn_and_cont(exn):
            """Called in an exception handler to ignore any exception, isssue a warning, and continue."""
            warnings.warn(repr(exn))
            return True

        shuffle_buffer = args.shuffle_buffer if phase == 'train' else 10

        def augment(src):
            return transform_openimages(src, aug=transform)

        def none_filter(src):  # TODO: check
            return filter_nones(src, has_keys=['image.jpg', 'targets.json'])

        dataset = (wds.Dataset(urls,
                               length=None,
                               # tarhandler=None,
                               tarhandler=warn_and_cont
                               )
                   # .pipe(none_filter)
                   .pipe(decode_openimages)
                   .pipe(augment)  # still in image, mask, target format; bboxes are numpy vectors
                   .pipe(shuffle(shuffle_buffer, initial=100, rng=utils.NumpyRNG()))
                   .pipe(batched(batchsize=args.batch_size, partial=True, collation_fn=collate_openimages))
                   )
        # nominal = args.nominal_dataset_len // args.batch_size // (args.tpu_cores or args.gpus)
        # dataset = wds.ResizedDataset(dataset, length=5000000, nominal=nominal)  # no need for ResizedDataset if Multidataset

    if 1:  # basic DataLoader
        def collate_id(x_): return x_

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # because batching done in dataset
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=collate_id,
            pin_memory=args.tpu_cores is None,
            worker_init_fn=worker_init_fn,
        )
    if 0:  # try to get multidataset working, because then no need to worry about dataset size

        def unbatch_dct(data):
            """
            TODO: shit not all tensors in dict have batch dim :/
            :param data: MultiDatasetIterator with (x_batch, y_batch=dict)
            :yield: one example pair (x, y)
            """
            for sample in data:
                assert isinstance(sample, (tuple, list)), sample
                assert len(sample) > 0
                for i in range(len(sample[0])):
                    yield sample[0][i], {key: val[i] for key, val in sample[1].items()}

        loader = (wds.MultiDataset(dataset,
                                   workers=args.num_workers if phase == 'train' else 1,  # 1 for val because OOM easily
                                   pin_memory=False,  # problems with dict targets
                                   output_size=10000)
                  # .pipe(unbatch_dct)
                  # .pipe(shuffle(shuffle_buffer, initial=shuffle_buffer, rng=NumpyRNG()))
                  # .pipe(batched(batchsize=args.batch_size, partial=True, collation_fn=recollate_ytbb))
                  )

    return loader



def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def load_bvecs_data(fname, start_idx, end_idx):
    """Based on
    https://github.com/milvus-io/bootcamp/blob/15c87d5bd91c4a00c5bf103fdde15e9afed6aec5/solutions/partition_hybrid_search/partition_import.py

    Parameters
    ----------
    fname : filename
    start_idx : start index
    end_idx :

    Returns
    -------

    """
    # print(fname, ": ", begin_num )
    x = np.memmap(fname, dtype="uint8", mode="r")
    d = x[:4].view("int32")[0]  # dimension
    data = np.array(x.reshape(-1, d + 4)[start_idx:end_idx, 4:])
    return data



