import json
import numpy as np
import onnxruntime

from datasets import *
import pdb
import albumentations as alb
from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import DataLoader

import base64
import json

MODEL_PATH = 'model/deephash_imagenet_v1.onnx'
ort_session = onnxruntime.InferenceSession(MODEL_PATH)

def to_numpy(tensor):  # inputs to ONNX models are numpy arrays:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def _calc_vector_batch(img_batch):
    ort_inputs = {ort_session.get_inputs()[0].name: img_batch}
    ort_outs = ort_session.run(None, ort_inputs)  # OK outputs actually list instead of dict...
    vectors = []
    for latents_ort in ort_outs[0]:
        hash_code = list(((np.sign(latents_ort) + 1) / 2).astype(np.uint8))
        # convert to bytes for binary index
        vector = bytes([int(''.join(map(str, hash_code[i:i+8])), 2) for i in range(0, len(hash_code), 8)])
        vectors.append(vector)

    return vectors


# augs = alb.Compose([alb.SmallestMaxSize(224),
#                 alb.CenterCrop(224, 224),
#                 ToTensor()])


def _calc_vector_batch_float(img_batch):
    ort_inputs = {ort_session.get_inputs()[0].name: img_batch}
    ort_outs = ort_session.run(None, ort_inputs)  # OK outputs actually list instead of dict...
    return ort_outs[0]


def _transform_vector_2bytes(vector):
    hash_code = list(((np.sign(vector) + 1) / 2).astype(np.uint8))
    # convert to bytes for binary index
    return bytes([int(''.join(map(str, hash_code[i:i+8])), 2) for i in range(0, len(hash_code), 8)])


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

augs = alb.Compose([alb.SmallestMaxSize(224),
                alb.CenterCrop(224, 224),
                ToTensor()])

s3_urls = ["https://imdatasets.s3.amazonaws.com/imagenet/train/n01440764/n01440764_10026.JPEG", "https://imdatasets.s3.amazonaws.com/imagenet/train/n01440764/n01440764_10027.JPEG", "https://imdatasets.s3.amazonaws.com/imagenet/train/n01440764/n01440764_10029.JPEG"]
ds = DataLoader(URLDataset(path_list=s3_urls, target_list=s3_urls, transform=augs, use_torchvision=False), num_workers=2)
result = []
for i, item in enumerate(ds):
     print(i)
     n = to_numpy(item[0])[0][0]
     print(n.shape)
     result.append(n)
print(len(result))
print(result[0].shape)
vectors = _calc_vector_batch(result)
print(vectors)

vectors = _calc_vector_batch_float(result)
results = []
for image_url, vector in zip(s3_urls, vectors):
    bvector = _transform_vector_2bytes(vector)
    results.append({"url": image_url, "vector": list(vector), "bvector": base64.b64encode(bvector).decode('utf-8')})

json_output = json.dumps(results, cls=NumpyEncoder)
print(json_output)
