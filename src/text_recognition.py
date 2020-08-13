from openvino.inference_engine import IECore
from image_iterator import ImageIterator
from config import *
import numpy as np
import cv2


class TextRecognizer:

    def __init__(self):
        self.ie = IECore()
        self.net = self.ie.read_network(model=rec_model_xml, weights=rec_model_bin)
        self.net.batch_size = 1
        self.exec_net = self.ie.load_network(network=self.net, num_requests=2, device_name='CPU')
        self.n, self.c, self.h, self.w = self.net.inputs['Placeholder'].shape
        self.symbols = "0123456789abcdefghijklmnopqrstuvwxyz#"

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def process(self, images: list, is_async_mode=True):
        texts, probabilities = list(), list()
        cur_request_id = 0
        next_request_id = 1
        images = ImageIterator(images=images)

        while images.isOpened():
            if is_async_mode:
                ret, next_image = images.read()
            else:
                ret, image = images.read()

            if not ret:
                break

            if is_async_mode:
                request_id = next_request_id
                input_image = cv2.cvtColor(next_image.astype(np.float32) * 255, cv2.COLOR_BGR2GRAY)
            else:
                request_id = cur_request_id
                input_image = cv2.cvtColor(image.astype(np.float32) * 255, cv2.COLOR_BGR2GRAY)

            input_image = cv2.resize(input_image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            input_image = input_image.reshape((self.n, self.c, self.h, self.w))
            self.exec_net.start_async(request_id=request_id, inputs={'Placeholder': input_image})

            if self.exec_net.requests[cur_request_id].wait(-1) == 0:
                outputs = self.exec_net.requests[cur_request_id].outputs['shadow/LSTMLayers/transpose_time_major']
                text = str()
                for symbol_id in outputs.argmax(2).flatten():
                    symbol = self.symbols[symbol_id]
                    text += symbol
                texts.append(text.replace('#', ''))

            if is_async_mode:
                cur_request_id, next_request_id = next_request_id, cur_request_id
                image = next_image

        return texts
