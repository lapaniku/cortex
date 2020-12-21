import json
import numpy as np
import onnxruntime
from milvus import Milvus, DataType
import redis
from google.cloud import storage
import io
from PIL import Image
import albumentations as alb
from albumentations.pytorch.transforms import ToTensor
import os
import boto3
import base64
import re

from datasets import URLDataset
from torch.utils.data import DataLoader


MODEL_PATH = 'model/deephash_imagenet_v1.onnx'


def to_numpy(tensor):  # inputs to ONNX models are numpy arrays:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_img_from_gcs(pth):
    """Load an image from GCS given relative path.

    :param pth: E.g. b'imagenet/train/n15075141/n15075141_9993.JPEG'
    :return:
    """

    store = storage.Client()
    bucket = store.bucket('imidatasets')
    blob = bucket.blob(pth)
    img_bytes = blob.download_as_string()
    stream = io.BytesIO(img_bytes)
    img = np.array(Image.open(stream).convert('RGB'))

    return img


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


class PythonPredictor:
    def __init__(self, config, job_spec):
        
        # self.redis_conn = redis.Redis(_HOST)
        # milvus = Milvus(_HOST, _PORT, pool="QueuePool")
        # self.milvus = Milvus(_HOST, pool="QueuePool", handler="HTTP")

        # status, ok = self.milvus.has_collection(collection_name)
        # if not ok:
        #     param = {
        #         'collection_name': collection_name,
        #         'dimension': 64,
        #         'metric_type': MetricType.HAMMING 
        #     }
        #     self.milvus.create_collection(param)

        
        # self.ort_session = onnxruntime.InferenceSession(MODEL_PATH)

        self.augs = alb.Compose([alb.SmallestMaxSize(224),
                        alb.CenterCrop(224, 224),
                        ToTensor()])

        # if len(config.get("dest_s3_dir", "")) == 0:
        #     raise Exception("'dest_s3_dir' field was not provided in job submission")

        # self.s3 = boto3.client("s3")

        # self.bucket, self.key = re.match("s3://(.+?)/(.+)", config["dest_s3_dir"]).groups()
        # self.key = os.path.join(self.key, job_spec["job_id"])


    def _calc_vector_batch_float(self, img_batch):
        ort_inputs = {self.ort_session.get_inputs()[0].name: img_batch}
        ort_outs = self.ort_session.run(None, ort_inputs)  # OK outputs actually list instead of dict...
        return ort_outs[0]


    def _transform_vector_2bytes(self, vector):
        hash_code = list(((np.sign(vector) + 1) / 2).astype(np.uint8))
        # convert to bytes for binary index
        return bytes([int(''.join(map(str, hash_code[i:i+8])), 2) for i in range(0, len(hash_code), 8)])


    def _augment(self, img):
        # for profiling
        x = self.augs(image=img)['image']
        return x[None]  # shape [3, 224, 224], values in [0...1]


    def predict(self, payload, batch_id):        
        try:
            s3_urls = ["https://imdatasets.s3.amazonaws.com/"+p for p in payload]
            ds = DataLoader(URLDataset(path_list=s3_urls, target_list=s3_urls, transform=self.augs, use_torchvision=False), num_workers=2)
            batch = []
            for item in ds:
                n = to_numpy(item[0])[0][0]
                batch.append(n)
            results = []
            # vectors = self._calc_vector_batch_float(batch)
            # for image_url, vector in zip(payload, vectors):
            #     bvector = self._transform_vector_2bytes(vector)
            #     results.append({"url": image_url, "vector": vector, "bvector": base64.b64encode(bvector).decode('utf-8')})

            # json_output = json.dumps(results, cls=NumpyEncoder)
            # self.s3.put_object(Bucket=self.bucket, Key=f"{self.key}/{batch_id}.json", Body=json_output)

            # status, ids = self.milvus.insert(collection_name=collection_name, records=vectors)
            # redis_conn.mset(dict(zip(ids, payload)))
            # in this case we never know which file corresponds to vector
        except redis.RedisError:
            logger.error('load strategys occur redis conn error')
        except Exception as e:
            raise Exception('ProcessingError')


    def on_job_complete(self):
        #not required yet
        # show how many records were saved in Milvus/Redis
        # or save vectors to Milvus/Redis
        pass

# if __name__ == '__main__':
#     predictor = PythonPredictor({'dest_s3_dir':'s3://bucket/dest'}, {"job_id":"123"})
#     predictor.predict(["1", "2", "3"], "123")
