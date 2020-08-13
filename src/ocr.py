import cv2
from config import *
from text_detection import TextDetector
from text_recognition import TextRecognizer


class OCR:

    def __init__(self):
        self.td = TextDetector()
        self.tr = TextRecognizer()

    def process_images(self, images=None, images_path=None):
        out = list()
        start = time.time()

        if images_path:
            images = [cv2.imread(image_path) for image_path in images_path]

        for boxes, cut_text_images in self.td.process(images=images):
            texts = self.tr.process(images=cut_text_images, is_async_mode=True)
            out.append(list(zip(boxes, texts)))

        log.info('Images processing time: {0} sec.'.format(round(time.time() - start), 2))
        return out
