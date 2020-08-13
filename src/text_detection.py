from openvino.inference_engine import IECore
from image_iterator import ImageIterator
from pixel_link_mobilenet import *
from skimage import transform
from config import *
import numpy as np
import cv2


class TextDetector:

    def __init__(self):
        self.ie = IECore()
        self.net = self.ie.read_network(model=det_model_xml, weights=det_model_bin)
        self.net.batch_size = 1
        self.exec_net = self.ie.load_network(network=self.net, num_requests=2, device_name='CPU')
        self.n, self.c, self.h, self.w = self.net.inputs['Placeholder'].shape
        self.pcd = PixelLinkDecoder()
        self.transformer = transform.ProjectiveTransform()

    def transform_text_image(self, image, box):
        w = max(box[1][0] - box[0][0], box[2][0] - box[3][0])
        h = max(box[2][1] - box[1][1], box[3][1] - box[0][1])
        src = np.array([[0, 0], [0, h], [w, h], [w, 0]])
        dst = np.array([box[0], box[3], box[2], box[1]])
        self.transformer.estimate(src, dst)
        warped = transform.warp(image, self.transformer, output_shape=(h, w))
        return warped

    def process(self, images: list):
        cur_request_id = 0
        next_request_id = 1
        first_image = True
        images = ImageIterator(images=images)
        last_image = None

        while images.isOpened():
            text_images = list()
            ret, next_image = images.read()

            if not ret:
                break

            last_image = next_image

            image_height, image_width = next_image.shape[:2]
            input_image = cv2.resize(next_image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

            input_image = input_image.transpose((2, 0, 1))
            input_image = input_image.reshape((self.n, self.c, self.h, self.w)).astype(np.float32)
            self.exec_net.start_async(request_id=next_request_id, inputs={'Placeholder': input_image})

            if not first_image:
                if self.exec_net.requests[cur_request_id].wait(-1) == 0:
                    outputs = self.exec_net.requests[cur_request_id].outputs
                    self.pcd.decode(image_height, image_width, outputs)
                    for box in self.pcd.bboxes:
                        text_images.append(self.transform_text_image(image=next_image, box=box))

            cur_request_id, next_request_id = next_request_id, cur_request_id
            first_image = False

            if text_images:
                yield self.pcd.bboxes, text_images

        # Getting last image
        text_images = list()
        if self.exec_net.requests[cur_request_id].wait(-1) == 0:
            image_height, image_width = last_image.shape[:2]
            outputs = self.exec_net.requests[cur_request_id].outputs
            self.pcd.decode(image_height, image_width, outputs)
            for box in self.pcd.bboxes:
                text_images.append(self.transform_text_image(image=last_image, box=box))
        if text_images:
            yield self.pcd.bboxes, text_images
