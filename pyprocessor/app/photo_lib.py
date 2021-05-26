#!/usr/bin/python3
import base64
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import grpc
import implicit
import numpy as np
import requests
from PIL import Image
from scipy import sparse
from sklearn.cluster import DBSCAN
from tensorflow import make_ndarray, make_tensor_proto
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

Image.MAX_IMAGE_PIXELS = None


def file2base64(path):
    with open(path, mode="rb") as fl:
        encoded = base64.b64encode(fl.read()).decode("ascii")
        return encoded


# Use https://github.com/SthPhoenix/InsightFace-REST to get faces
# TODO: Optimizaton make this do multiple items at once
def extract_vecs(image_file, max_size=[640, 480]):
    target = [file2base64(image_file)]
    req = {"images": {"data": target}, "max_size": max_size}
    resp = requests.post("http://localhost:18081/extract", json=req)
    data = resp.json()
    return data


def bbox_format(bbox):
    left, top, right, bottom = bbox
    return [top, left, bottom, right]


def get_bboxes(result):
    return [bbox_format(face["bbox"]) for face in result[0]]


def get_encodings(result):
    return [np.array(face["vec"]) for face in result[0]]


# TODO: Check if the accuracy improves if we scale the image based on exif rotation
class Detector:
    def __init__(
        self,
        width,
        height,
        grpc_address="localhost",
        grpc_port=9000,
    ):
        self.channel = grpc.insecure_channel(f"{grpc_address}:{grpc_port}")
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.height = height
        self.width = width
        self.model_name = "person-detection"

    def load_image(self, file_path, width, height):
        img = cv2.imread(file_path)  # BGR color format, shape HWC
        # print(file_path, img.shape)
        img_h, img_w, _c = img.shape
        imgs = cv2.resize(img, (width, height))
        # change shape to NCHW
        imgs = imgs.transpose(2, 0, 1).reshape(1, 3, height, width)
        return imgs, img_h, img_w, img

    def solo_predict(self, filename):
        # print("Start predict")
        imgs, img_h, img_w, img = self.load_image(filename, self.width, self.height)
        imgs = np.array(imgs, dtype=("<f"))

        # print("\nRequest shaper", imgs.shape)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        # print(request)
        request.inputs["data"].CopyFrom(make_tensor_proto(imgs, shape=(imgs.shape)))
        # start_time = datetime.datetime.now()
        # result includes a dictionary with all model outputs
        result = self.stub.Predict(request, 10.0)
        # end_time = datetime.datetime.now()
        # duration = (end_time - start_time).total_seconds() * 1000
        duration = 0
        output = make_ndarray(result.outputs["detection_out"])
        # print("Response shape", output.shape)

        bboxes = []
        embeddings = []
        for i in range(0, 200 - 1):
            detection = output[:, :, i, :]
            # each detection has shape 1,1,7 where last dimension represent:
            # image_id - ID of the image in the batch
            # label - predicted class ID
            # conf - confidence for the predicted class
            # (x_min, y_min) - coordinates of the top left bounding box corner
            # (x_max, y_max) - coordinates of the bottom right bounding box corner.

            # ignore detections for image_id != y and confidence <0.5
            label = int(detection[0, 0, 0])
            confidence = detection[0, 0, 2]
            if confidence > 0.5 and label == 0:
                # print("detection", i, detection)
                bbox = self.get_person_bboxes(detection, img_w, img_h)

                bboxes.append(bbox)
                # print("detection", i, bbox)
                embedding = self.get_embedding(img, bbox)  # note: img, not imgs
                embeddings.append(embedding)
        return duration, output, bboxes, embeddings

    def get_embedding(self, input_image, bbox):
        # print("Start get Embedding", input_image.shape)

        x_min, y_min, x_max, y_max = bbox
        # y_min, x_min, y_max, x_max = bbox

        img_bbox_out = input_image[y_min:y_max, x_min:x_max]
        print(img_bbox_out.shape, input_image.shape, bbox)
        cv2.imwrite("person.jpg", img_bbox_out)

        # Model requires input in a particular size
        width = 128
        height = 256
        model_name = "person-reidentification"

        img = cv2.resize(img_bbox_out, (width, height))
        # change shape to NCHW
        img = img.transpose(2, 0, 1).reshape(1, 3, height, width)
        imgs = np.array(img, dtype=("<f"))

        # print("\nRequest shaper", imgs.shape)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        # print(request)
        request.inputs["data"].CopyFrom(make_tensor_proto(imgs, shape=(imgs.shape)))

        start_time = datetime.now()
        result = self.stub.Predict(request, 10.0)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000
        time.sleep(0.5)  # TODO: Without this, MY system crashes; remove in production
        # print(duration)

        output = make_ndarray(result.outputs["reid_embedding"])
        output = np.squeeze(output).tolist()  # delete axis with length 1
        # print("Response shape", len(output), duration)
        return output

    @staticmethod
    def get_person_bboxes(detection, width, height):
        # each detection has shape 1,1,7 where last dimension represent:
        # image_id - ID of the image in the batch
        # label - predicted class ID
        # conf - confidence for the predicted class
        # (x_min, y_min) - coordinates of the top left bounding box corner
        # (x_max, y_max) - coordinates of the bottom right bounding box corner.
        x_min = int(detection[0, 0, 3] * width)
        y_min = int(detection[0, 0, 4] * height)
        x_max = int(detection[0, 0, 5] * width)
        y_max = int(detection[0, 0, 6] * height)

        x_min, y_min, x_max, y_max = Detector.bbox_check(
            width, height, x_min, y_min, x_max, y_max
        )
        return x_min, y_min, x_max, y_max

    @staticmethod
    def bbox_check(width, height, x_min, y_min, x_max, y_max):
        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if x_max > width:
            x_max = width
        if y_max > height:
            y_max = height
        print(x_min, y_min, x_max, y_max)
        assert x_min >= 0
        assert y_min >= 0
        assert y_max >= 0
        assert y_max >= 0
        assert y_max <= height
        assert x_max <= width
        return x_min, y_min, x_max, y_max


detector = Detector(600, 400)


@dataclass
class Photofile:
    filename: Path
    face_locations: list
    person_locations: list
    embeddings: list
    person_embeddings: list
    processed: bool
    face_indices: list
    person_indices: list

    def get_face_locations(self):

        if self.face_locations != []:
            self.processed = True
            # print(f"embedding exists for {self.filename.stem}")
            # self.get_person_locations()
            return self.face_locations
        # Load the jpg file into a numpy array
        # print(f"generating embedding for {self.filename}")

        result = extract_vecs(self.filename)
        self.face_locations = get_bboxes(result)
        self.face_encodings = get_encodings(result)
        # self.get_person_locations()
        # print(self.face_encodings[0].shape)

        print(
            f"I found {len(self.face_locations)} face(s)  and {len(self.person_locations)} person(s) in this photograph:{self.filename.stem}"
        )

        self.processed = True
        return self.face_locations

    def get_person_locations(self):
        # print("Get person locations")

        if self.person_locations != []:
            self.processed = True
            # print(f"person location exists for {self.filename}")
            return self.person_locations

        (
            _duration,
            _output,
            self.person_locations,
            self.person_embeddings,
        ) = detector.solo_predict(str(self.filename))
        print(self.person_locations)

        self.processed = True
        return self.person_locations

    def set_face_indices(self, indices):
        self.face_indices = indices

    def set_person_indices(self, indices):
        self.person_indices = indices

    @staticmethod
    def bbox_modify(bbox):
        left, top, right, bottom = bbox
        return top, left, bottom, right


class PhotoDict:
    def __init__(self, fileList):
        self.filelist = fileList
        self.dict = {}
        pickle_file = "databp.pickle"
        self.faces = {}

        if Path(pickle_file).is_file():
            print("Loading Cached embeddings and locations")
            with open(pickle_file, "rb") as file_to_read:
                self.dict = pickle.load(file_to_read)
                keys = self.dict.keys()
                old_keys = [item for item in keys if item not in fileList]
                for key in old_keys:
                    print(f"Removing {key} data from cache")
                    del self.dict[key]
                # print(self.dict.keys())

        for file in self.filelist:
            if self.dict.get(file) is None:
                photo_file = Photofile(file, [], [], [], [], False, [], [])
                photo_file.get_face_locations()
                self.dict[file] = photo_file

        with open(pickle_file, "wb") as file_to_write:
            pickle.dump(self.dict, file_to_write)

        # # clustering = DBSCAN(eps=0.4, min_samples=3).fit(self.encodings) # facenet
        # # clustering = DBSCAN(eps=0.8, min_samples=3).fit(self.encodings)  # insightface
        # clustering = DBSCAN(eps=2.3, min_samples=2).fit(self.encodings)  # person-reid

        self.face_labels = self.get_face_labels()
        # self.person_labels = self.get_person_labels()
        self.person_labels = self.face_labels

        print(Counter(self.face_labels))
        self.graph = self.bipartite_graph()
        self.model = None  # to prevent unknown attribute error
        # print(self.sort_by_userid(7))

    def sort_by_userid(self, userid):

        # desired shape of  (number_of_items, number_of_users)
        item_users = sparse.csr_matrix(self.graph.T)

        if self.model is None:
            model = implicit.als.AlternatingLeastSquares(factors=50)
            # model = implicit.bpr.BayesianPersonalizedRanking(iterations=1000)
            model.fit(item_users)
            self.model = model

        # recommend items for a user
        user_items = item_users.T.tocsr()
        recommendations = self.model.recommend(
            userid, user_items, filter_already_liked_items=False, N=len(self.filelist)
        )

        return [self.filelist[idx] for (idx, _) in recommendations]

    def bipartite_graph(self):
        # total_faces = len(self.face_labels)
        unique_faces = len(set(self.face_labels))
        total_photos = len(self.filelist)

        graph = np.zeros((unique_faces - 1, total_photos))
        print(graph.shape)

        for index, file in enumerate(self.filelist):
            indices = self.dict.get(file).face_indices
            # because we only want interesting faces, remove -1
            face_labels = [self.face_labels[f] for f in indices if f != -1]
            # print(file, index, sorted(face_labels))
            for face_label in face_labels:
                graph[face_label, index] = 1
                # if face_label != -1:
            # print(np.transpose(np.argwhere(graph[:, index])))
        # print(np.count_nonzero(graph), total_faces)
        # print(np.where(graph == 1).shape)

        # graph = graph[:-1, :]
        # print(graph.shape)
        return graph

    def get_person_labels(self):
        all_arr1 = np.array([])
        print(all_arr1.shape)
        firstIteration = True
        for file in self.filelist:

            self.faces[file] = self.dict.get(file).person_embeddings
            if self.faces[file] != []:
                # print(file)
                np_array = np.concatenate(
                    [np.expand_dims(c, axis=0) for c in self.faces[file]], axis=0
                )

                if firstIteration:
                    np_length = 0
                    all_arr1 = np_array
                    firstIteration = False
                else:
                    np_length, _ = all_arr1.shape
                    all_arr1 = np.concatenate([all_arr1, np_array], axis=0)

                indices = [np_length + idx for idx, _ in enumerate(self.faces[file])]
                self.dict[file].set_person_indices(indices)
        print(all_arr1.shape)
        clustering = DBSCAN(eps=2.5, min_samples=2, metric="euclidean").fit(
            all_arr1
        )  # person-reid
        # print(set(clustering.labels_), len(clustering.labels_))
        person_labels = clustering.labels_
        return person_labels

    def get_face_labels(self):
        all_arr1 = np.array([])
        print(all_arr1.shape)
        firstIteration = True
        for file in self.filelist:
            self.faces[file] = self.dict.get(file).face_encodings

            if self.faces[file] != []:
                # print(file)
                np_array = np.concatenate(
                    [np.expand_dims(c, axis=0) for c in self.faces[file]], axis=0
                )

                if firstIteration:
                    np_length = 0
                    all_arr1 = np_array
                    firstIteration = False
                else:
                    np_length, _ = all_arr1.shape
                    all_arr1 = np.concatenate([all_arr1, np_array], axis=0)

                indices = [np_length + idx for idx, _ in enumerate(self.faces[file])]
                self.dict[file].set_face_indices(indices)
        print(all_arr1.shape)

        clustering = DBSCAN(eps=0.7, min_samples=4).fit(all_arr1)  # insight-face
        print(set(clustering.labels_), len(clustering.labels_))
        face_labels = clustering.labels_
        return face_labels

    def get_photo_file(self, filename):
        return self.dict[filename]


def main():
    print("This is a library. Do not call this directly")


if __name__ == "__main__":
    main()
