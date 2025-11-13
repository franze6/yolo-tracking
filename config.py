"""
Конфигурационные параметры для детекции и отслеживания объектов.
"""

# Параметры детекции
CONF_THRESHOLD = 0.5
MODEL_PATH = "models/yolo11m.pt"

# Параметры DeepSORT
MAX_COSINE_DISTANCE = 0.4
NN_BUDGET = None
DEEPSORT_MODEL_PATH = "models/mars-small128.pb"
CLASSES_PATH = "config/coco.names"

# Параметры предобработки кадров
PREPROCESS_ENABLED = True
PREPROCESS_CLAHE = True
PREPROCESS_DENOISING = False
PREPROCESS_SHARPENING = True
PREPROCESS_BRIGHTNESS_NORMALIZATION = True

# Параметры CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Параметры шумоподавления
DENOISE_H = 10
DENOISE_TEMPLATE_WINDOW_SIZE = 7
DENOISE_SEARCH_WINDOW_SIZE = 21

# Параметры увеличения резкости
SHARPEN_STRENGTH = 0.5

# Параметры ESRGAN upscaling
PREPROCESS_UPSCALE = False  # Включить upscaling с помощью ESRGAN
UPSCALE_FACTOR = 1.5  # Коэффициент увеличения (2x, 4x). Должен соответствовать модели (x4plus = 4, x2plus = 2)
UPSCALE_MODEL = 'models/RealESRGAN_x4plus.pth'  # Путь к модели (.pth файл) или название модели для автоматической загрузки
# ВАЖНО: При включении upscaling необходимо масштабировать координаты линий подсчета (START_LINE_A, END_LINE_A, START_LINE_B, END_LINE_B) на UPSCALE_FACTOR

# Параметры многопроцессности
USE_MULTIPROCESSING = True
from multiprocessing import cpu_count
NUM_WORKERS = max(1, cpu_count() - 1)  # Количество потоков для предобработки
BATCH_SIZE = 16  # Размер батча для YOLO

# Параметры линий подсчета
START_LINE_A = (290, 317)
END_LINE_A = (607, 297)
START_LINE_B = (726, 383)
END_LINE_B = (1036, 343)

# Параметры трекинга
MAX_TRACK_POINTS = 32  # Максимальное количество точек для отслеживания траектории
MAX_TRACK_ID = 1000  # Максимальный ID трека

# Параметры видео
INPUT_VIDEO = "1.mp4"
OUTPUT_VIDEO = "output.mp4"
DEFAULT_FPS = 30  # FPS по умолчанию, если не определен в видео

# Классы для исключения из отслеживания
EXCLUDED_CLASSES = ["person", "traffic light"]

