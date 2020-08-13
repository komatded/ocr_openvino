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
        self.net = self.ie.read_network(model=model_xml, weights=model_bin)
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

    def predict(self, images: list, is_async_mode=True):
        text_images = list()
        cur_request_id = 0
        next_request_id = 1
        images = ImageIterator(images=images)

        while images.isOpened():
            if is_async_mode:
                ret, next_image = images.read()
                orig_image = next_image
            else:
                ret, image = images.read()
                orig_image = image

            if not ret:
                break

            image_height, image_width = orig_image.shape[:2]

            if is_async_mode:
                request_id = next_request_id
                input_image = cv2.resize(next_image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            else:
                request_id = cur_request_id
                input_image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

            input_image_size = input_image.shape[:2]
            input_image = np.pad(input_image, ((0, self.h - input_image_size[0]),
                                               (0, self.w - input_image_size[1]),
                                               (0, 0)),
                                 mode='constant', constant_values=0)
            input_image = input_image.transpose((2, 0, 1))
            input_image = input_image.reshape((self.n, self.c, self.h, self.w)).astype(np.float32)
            self.exec_net.start_async(request_id=request_id, inputs={'Placeholder': input_image})

            if self.exec_net.requests[cur_request_id].wait(-1) == 0:
                outputs = self.exec_net.requests[cur_request_id].outputs
                self.pcd.decode(image_height, image_width, outputs)
                for box in self.pcd.bboxes:
                    text_images.append(self.transform_text_image(image=orig_image, box=box))

            if is_async_mode:
                cur_request_id, next_request_id = next_request_id, cur_request_id
                image = next_image

        return text_images

