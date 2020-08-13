import cv2
import json
import base64
import numpy as np
from ocr import OCR
from aiohttp import web
from mrrest import RESTApi
from config import log

ocr = OCR()


def load_images(images: list):
    out = list()
    for image_encoded in images:
        image_bytes = base64.b64decode(image_encoded)
        image_array = np.frombuffer(image_bytes, np.uint8)
        out.append(cv2.imdecode(image_array, cv2.IMREAD_COLOR))
    return out


async def test(request: web.Request):
    try:
        request = await request.json()
    except json.JSONDecodeError:
        raise web.HTTPBadRequest(text='wrong json format')

    images = load_images(request['images'])
    log.info('Loaded images: {0}'.format(len(images)))
    result = ocr.process_images(images=images)
    return result


api = RESTApi(
    host='0.0.0.0',
    port=8000,
    routes=[web.post('/test', test)])

api.app._client_max_size = 1e+8
api.run()


# if __name__ == '__main__':
#     import time
#     ocr = OCR()
#     images = [cv2.imread(test_image)] * 100
#     start = time.time()
#     result = ocr.process_images(images=images)
#     duration = time.time() - start
#     log.info('Full time: {0} sec, {1} sec per image'.format(round(duration, 2), round(duration / len(images), 2)))
#     log.info('Images processed: {0}'.format(len(result)))
#     log.info('Test result: {0}'.format(result[-1]))
