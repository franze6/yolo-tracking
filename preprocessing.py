"""
Модуль для предобработки кадров видео.
"""

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import time
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
ESRGAN_AVAILABLE = True



# Глобальный словарь для хранения моделей ESRGAN (ленивая загрузка)
# Ключ: (model_name, scale), значение: модель
_esrgan_models = {}


def load_esrgan_model(model_name='realesrgan-x4plus', scale=4):
    """
    Загружает модель ESRGAN для upscaling.
    
    Parameters:
    -----------
    model_name : str
        Название модели ('realesrgan-x4plus', 'realesrgan-x4plus-anime', 'realesrnet-x4plus')
    scale : int
        Коэффициент увеличения (2 или 4)
    
    Returns:
    --------
    model : RealESRGANer или None
        Загруженная модель или None если не удалось загрузить
    """
    global _esrgan_models
    
    if not ESRGAN_AVAILABLE:
        return None
    
    # Проверяем, загружена ли уже модель с такими параметрами
    model_key = (model_name, scale)
    if model_key in _esrgan_models:
        return _esrgan_models[model_key]
    
    try:
        import os
        import torch
        
        # Проверяем, является ли model_name путем к локальному файлу
        is_local_file = model_name.endswith('.pth') and os.path.exists(model_name)
        
        if is_local_file:
            # Загружаем checkpoint, чтобы определить параметры модели
            state_dict = torch.load(model_name, map_location=torch.device('cuda'))['params_ema']
            
            # Определяем num_in_ch из первого слоя
            # Ищем ключ conv_first.weight (может быть с префиксом или без)
            num_in_ch = 3  # По умолчанию для Real-ESRGAN
            first_conv_key = None
            for key in state_dict.keys():
                if 'conv_first.weight' in key:
                    first_conv_key = key
                    num_in_ch = state_dict[key].shape[1]
                    break
            
            if first_conv_key:
                print(f"Определен num_in_ch={num_in_ch} из ключа '{first_conv_key}'")
            else:
                print(f"Не удалось найти conv_first.weight в checkpoint, используется num_in_ch={num_in_ch} по умолчанию")
            
            # Определяем архитектуру модели на основе имени файла и параметров
            if 'anime' in model_name.lower():
                model = RRDBNet(num_in_ch=num_in_ch, num_out_ch=3, num_feat=64, 
                               num_block=6, num_grow_ch=32, scale=scale)
            else:
                # Стандартная модель Real-ESRGAN x4plus
                model = RRDBNet(num_in_ch=num_in_ch, num_out_ch=3, num_feat=64, 
                               num_block=23, num_grow_ch=32, scale=scale)

            model.load_state_dict(state_dict, strict=True)
            
            # Создаем upsampler с правильной моделью
            upsampler = RealESRGANer(
                scale=scale,
                model_path=model_name,  # Путь к локальному файлу
                model=model,  # Передаем созданную модель
                tile=0,  # Размер тайла для обработки больших изображений (0 = без тайлов)
                pre_pad=0,
                half=False  # Использовать float32 (True для float16, быстрее но менее точно)
            )
        else:
            # Если указано имя модели (не путь), создаем архитектуру вручную
            # Определяем архитектуру модели в зависимости от названия
            if 'anime' in model_name.lower():
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                               num_block=6, num_grow_ch=32, scale=scale)
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                               num_block=23, num_grow_ch=32, scale=scale)
            
            # Создаем upsampler с указанием модели по имени
            # Real-ESRGAN автоматически загрузит модель по имени
            upsampler = RealESRGANer(
                scale=scale,
                model_path=model_name,  # Название модели для автоматической загрузки
                model=model,
                tile=0,  # Размер тайла для обработки больших изображений (0 = без тайлов)
                pre_pad=0,
                half=True  # Использовать float32 (True для float16, быстрее но менее точно)
            )
        
        _esrgan_models[model_key] = upsampler
        print(f"Модель ESRGAN '{model_name}' (scale={scale}x) успешно загружена")
        return _esrgan_models[model_key]
    except Exception as e:
        print(f"Ошибка при загрузке модели ESRGAN: {e}")
        print("Убедитесь, что установлены библиотеки: pip install realesrgan basicsr")
        import traceback
        traceback.print_exc()
        return None


def upscale_frame(frame, upscale_factor=2, model_name='realesrgan-x4plus'):
    """
    Увеличивает разрешение кадра с помощью ESRGAN.
    
    Parameters:
    -----------
    frame : np.ndarray
        Входной кадр в формате BGR
    upscale_factor : int
        Коэффициент увеличения (2 или 4)
    model_name : str
        Название модели для upscaling
    
    Returns:
    --------
    upscaled_frame : np.ndarray
        Увеличенный кадр или оригинал, если upscaling недоступен
    """
    if not ESRGAN_AVAILABLE or not config.PREPROCESS_UPSCALE:
        return frame
    
    try:
        upsampler = load_esrgan_model(model_name, upscale_factor)
        if upsampler is None:
            return frame
        
        # Конвертируем BGR в RGB для Real-ESRGAN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Применяем upscaling
        # outscale=1, так как scale уже указан при создании модели
        output, _ = upsampler.enhance(frame_rgb, outscale=1)
        
        # Конвертируем обратно в BGR
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output_bgr
    except Exception as e:
        print(f"Ошибка при upscaling кадра: {e}")
        return frame

def preprocess_frame_without_upscale(frame,
                                     enable_clahe=True,
                                     enable_denoising=True,
                                     enable_sharpening=False,
                                     enable_brightness_normalization=True,
                                     clahe_clip_limit=2.0,
                                     clahe_tile_grid_size=(8, 8),
                                     denoise_h=10,
                                     denoise_template_window_size=7,
                                     denoise_search_window_size=21,
                                     sharpen_strength=0.5):
    """
    Предобработка кадра без upscaling (для параллельной обработки).
    
    Parameters:
    -----------
    frame : np.ndarray
        Входной кадр в формате BGR
    enable_clahe : bool
        Включить CLAHE (контрастная адаптивная гистограмма)
    enable_denoising : bool
        Включить шумоподавление
    enable_sharpening : bool
        Включить увеличение резкости
    enable_brightness_normalization : bool
        Включить нормализацию яркости
    clahe_clip_limit : float
        Лимит обрезки для CLAHE
    clahe_tile_grid_size : tuple
        Размер сетки для CLAHE
    denoise_h : int
        Параметр шумоподавления (цветовой компонент)
    denoise_template_window_size : int
        Размер окна шаблона для шумоподавления
    denoise_search_window_size : int
        Размер окна поиска для шумоподавления
    sharpen_strength : float
        Сила увеличения резкости (0.0 - 1.0)
    
    Returns:
    --------
    processed_frame : np.ndarray
        Обработанный кадр
    """
    processed = frame.copy()
    
    # Конвертация в LAB цветовое пространство для лучшей обработки яркости
    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # 1. Нормализация яркости (работаем с L-каналом)
    if enable_brightness_normalization:
        # Вычисляем среднюю яркость
        mean_brightness = np.mean(l_channel)
        target_brightness = 128  # целевая средняя яркость
        
        # Корректируем яркость если она слишком низкая или высокая
        if mean_brightness < 100 or mean_brightness > 160:
            brightness_diff = target_brightness - mean_brightness
            l_channel = cv2.add(l_channel, int(brightness_diff * 0.3))
            l_channel = np.clip(l_channel, 0, 255).astype(np.uint8)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if enable_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
        l_channel = clahe.apply(l_channel)
    
    # Объединяем каналы обратно
    lab = cv2.merge([l_channel, a, b])
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 3. Шумоподавление (быстрый алгоритм для видео)
    if enable_denoising:
        # Используем fastNlMeansDenoisingColored для цветных изображений
        processed = cv2.fastNlMeansDenoisingColored(
            processed,
            None,
            h=denoise_h,
            hColor=denoise_h,
            templateWindowSize=denoise_template_window_size,
            searchWindowSize=denoise_search_window_size
        )
    
    # 4. Увеличение резкости
    if enable_sharpening:
        # Создаем ядро для увеличения резкости
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]]) * sharpen_strength
        # Сглаженное изображение для смешивания
        smoothed = cv2.GaussianBlur(processed, (0, 0), 1.0)
        # Применяем увеличение резкости
        sharpened = cv2.filter2D(processed, -1, kernel)
        # Смешиваем оригинал и резкое изображение
        processed = cv2.addWeighted(processed, 1 - sharpen_strength, sharpened, sharpen_strength, 0)
        processed = np.clip(processed, 0, 255).astype(np.uint8)
    
    return processed


def preprocess_frames_parallel(frames, num_workers=None):
    """
    Параллельная предобработка кадров с использованием multiprocessing.
    Upscaling выполняется последовательно, остальные операции - параллельно.
    
    Parameters:
    -----------
    frames : list
        Список кадров для предобработки
    num_workers : int
        Количество процессов для обработки
    
    Returns:
    --------
    processed_frames : list
        Список обработанных кадров
    """
    if not config.PREPROCESS_ENABLED:
        return frames
    
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    
    total_frames = len(frames)
    start_time = time.time()
    
    # Определяем интервал для вывода прогресса (каждые 10% или минимум каждые 10 кадров)
    progress_interval = max(10, total_frames // 10)
    
    # Шаг 1: Upscaling (выполняется последовательно, если включен)
    if config.PREPROCESS_UPSCALE:
        print(f"Начало upscaling {total_frames} кадров (последовательно)...")
        upscaled_frames = []
        upscale_start_time = time.time()
        
        for idx, frame in enumerate(frames):
            upscaled_frame = upscale_frame(frame, config.UPSCALE_FACTOR, config.UPSCALE_MODEL)
            upscaled_frames.append(upscaled_frame)
            
            # Выводим прогресс upscaling
            if (idx + 1) % progress_interval == 0 or (idx + 1) == total_frames:
                elapsed_time = time.time() - upscale_start_time
                progress_percent = ((idx + 1) / total_frames) * 100
                fps = (idx + 1) / elapsed_time if elapsed_time > 0 else 0
                remaining_frames = total_frames - (idx + 1)
                eta = remaining_frames / fps if fps > 0 else 0
                
                print(f"Upscaling: {idx + 1}/{total_frames} кадров ({progress_percent:.1f}%) | "
                      f"Скорость: {fps:.2f} кадр/сек | "
                      f"Осталось: ~{eta:.1f} сек")
        
        upscale_time = time.time() - upscale_start_time
        print(f"Upscaling завершен: {total_frames} кадров обработано за {upscale_time:.2f} сек")
        frames_to_process = upscaled_frames
    else:
        frames_to_process = frames
    
    # Шаг 2: Остальные операции предобработки (параллельно)
    # Параметры предобработки (без upscaling)
    params = {
        'enable_clahe': config.PREPROCESS_CLAHE,
        'enable_denoising': config.PREPROCESS_DENOISING,
        'enable_sharpening': config.PREPROCESS_SHARPENING,
        'enable_brightness_normalization': config.PREPROCESS_BRIGHTNESS_NORMALIZATION,
        'clahe_clip_limit': config.CLAHE_CLIP_LIMIT,
        'clahe_tile_grid_size': config.CLAHE_TILE_GRID_SIZE,
        'denoise_h': config.DENOISE_H,
        'denoise_template_window_size': config.DENOISE_TEMPLATE_WINDOW_SIZE,
        'denoise_search_window_size': config.DENOISE_SEARCH_WINDOW_SIZE,
        'sharpen_strength': config.SHARPEN_STRENGTH
    }
    
    # Проверяем, есть ли другие операции предобработки
    has_other_ops = (config.PREPROCESS_CLAHE or config.PREPROCESS_DENOISING or 
                     config.PREPROCESS_SHARPENING or config.PREPROCESS_BRIGHTNESS_NORMALIZATION)
    
    if not has_other_ops:
        # Если нет других операций, просто возвращаем кадры после upscaling
        total_time = time.time() - start_time
        print(f"Предобработка завершена: {total_frames} кадров обработано за {total_time:.2f} сек")
        return frames_to_process
    
    print(f"Начало остальных операций предобработки {total_frames} кадров (используется {num_workers} потоков)")
    other_ops_start_time = time.time()
    
    # Используем ThreadPoolExecutor для параллельной обработки
    # OpenCV операции могут быть быстрее в потоках благодаря оптимизациям OpenCV
    if config.USE_MULTIPROCESSING and len(frames_to_process) > 1:
        # Параллельная обработка с сохранением порядка
        processed_frames = [None] * len(frames_to_process)  # Предварительно создаем список нужного размера
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Создаем словарь для отслеживания порядка
            future_to_index = {}
            
            # Отправляем все задачи на обработку
            for idx, frame in enumerate(frames_to_process):
                future = executor.submit(preprocess_frame_without_upscale, frame, **params)
                future_to_index[future] = idx
            
            # Собираем результаты в правильном порядке
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                processed_frames[idx] = future.result()
                completed_count += 1
                
                # Выводим прогресс
                if completed_count % progress_interval == 0 or completed_count == total_frames:
                    elapsed_time = time.time() - other_ops_start_time
                    progress_percent = (completed_count / total_frames) * 100
                    fps = completed_count / elapsed_time if elapsed_time > 0 else 0
                    remaining_frames = total_frames - completed_count
                    eta = remaining_frames / fps if fps > 0 else 0
                    
                    print(f"Предобработка: {completed_count}/{total_frames} кадров ({progress_percent:.1f}%) | "
                          f"Скорость: {fps:.2f} кадр/сек | "
                          f"Осталось: ~{eta:.1f} сек")
    else:
        # Последовательная обработка
        processed_frames = []
        for idx, frame in enumerate(frames_to_process):
            processed_frames.append(preprocess_frame_without_upscale(frame, **params))
            completed_count = idx + 1
            
            # Выводим прогресс
            if completed_count % progress_interval == 0 or completed_count == total_frames:
                elapsed_time = time.time() - other_ops_start_time
                progress_percent = (completed_count / total_frames) * 100
                fps = completed_count / elapsed_time if elapsed_time > 0 else 0
                remaining_frames = total_frames - completed_count
                eta = remaining_frames / fps if fps > 0 else 0
                
                print(f"Предобработка: {completed_count}/{total_frames} кадров ({progress_percent:.1f}%) | "
                      f"Скорость: {fps:.2f} кадр/сек | "
                      f"Осталось: ~{eta:.1f} сек")
    
    total_time = time.time() - start_time
    avg_fps = total_frames / total_time if total_time > 0 else 0
    print(f"Предобработка завершена: {total_frames} кадров обработано за {total_time:.2f} сек (средняя скорость: {avg_fps:.2f} кадр/сек)")
    
    return processed_frames

