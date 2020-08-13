import os
import time
import logging

# Логирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('urllib3.connection').setLevel(logging.CRITICAL)
log = logging.getLogger('ad_parser')

log_level = os.getenv('LOGLEVEL', 'DEBUG')
assert log_level in ('CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET')

log.setLevel(log_level)

# Переменные запуска
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCE_DIR = os.path.abspath(os.path.join(MAIN_DIR, 'resources'))

det_model_xml = os.path.join(RESOURCE_DIR, 'models/text-detection-0004/text-detection-0004.xml')
det_model_bin = os.path.join(RESOURCE_DIR, 'models/text-detection-0004/text-detection-0004.bin')
rec_model_xml = os.path.join(RESOURCE_DIR, 'models/text-recognition-0012/text-recognition-0012.xml')
rec_model_bin = os.path.join(RESOURCE_DIR, 'models/text-recognition-0012/text-recognition-0012.bin')

test_image = os.path.join(RESOURCE_DIR, 'test.jpg')
