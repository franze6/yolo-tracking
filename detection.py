"""
Модуль для детекции объектов с использованием YOLO.
"""

import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
import config


def load_model(model_path=None):
    """
    Загружает модель YOLO.
    
    Parameters:
    -----------
    model_path : str
        Путь к файлу модели YOLO
    
    Returns:
    --------
    model : YOLO
        Загруженная модель YOLO
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    return YOLO(model_path)


def process_frames_batch(model, frames, batch_size=None, device=0):
    """
    Обрабатывает кадры батчами с использованием YOLO.
    
    Parameters:
    -----------
    model : YOLO
        Модель YOLO
    frames : list
        Список кадров для обработки
    batch_size : int
        Размер батча
    device : int
        ID устройства (GPU)
    
    Returns:
    --------
    results : list
        Список результатов YOLO для каждого кадра
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    yolo_results = []
    
    # Обрабатываем кадры батчами для ускорения
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        batch_size_actual = len(batch)
        
        # YOLO может обрабатывать несколько кадров одновременно
        batch_results = model(batch, device=device, verbose=False)
        
        # YOLO возвращает объект Results, который можно итерировать
        # При batch обработке каждая итерация дает Results для одного кадра
        try:
            # Проверяем, является ли результат итерируемым (список Results объектов)
            if hasattr(batch_results, '__iter__'):
                # Преобразуем в список для надежности
                results_list = list(batch_results)
                
                # Проверяем количество результатов
                if len(results_list) != batch_size_actual:
                    print(f"Предупреждение: ожидалось {batch_size_actual} результатов, получено {len(results_list)}")
                
                # Добавляем каждый результат в список
                for idx, result in enumerate(results_list):
                    # Проверяем, что результат не пустой
                    if result is not None:
                        yolo_results.append(result)
                    else:
                        print(f"Предупреждение: результат {idx} в батче {i//batch_size + 1} равен None")
            else:
                # Если результат не итерируемый, это один кадр
                if batch_size_actual == 1:
                    yolo_results.append(batch_results)
                else:
                    print(f"Ошибка: батч размером {batch_size_actual} вернул неитерируемый результат")
                    # Пытаемся добавить как есть
                    yolo_results.append(batch_results)
        except Exception as e:
            # Если возникла ошибка при итерации
            print(f"Ошибка при обработке батча {i//batch_size + 1}: {e}")
            import traceback
            traceback.print_exc()
            # Пробуем добавить как есть
            yolo_results.append(batch_results)
        
        # Прогресс
        processed = min(i + batch_size, len(frames))
        if processed % (batch_size * 10) == 0 or processed == len(frames):
            print(f"Обработано {processed} / {len(frames)} кадров")
            print(f"  Результатов YOLO добавлено: {len(yolo_results)}")
    
    return yolo_results


def extract_detections_and_features(frame_idx, results, frame, encoder, class_names, conf_threshold):
    """
    Извлекает детекции из результатов YOLO и признаки для DeepSORT.
    Эта функция может выполняться параллельно для разных кадров.
    
    Parameters:
    -----------
    frame_idx : int
        Индекс кадра
    results : YOLO Results
        Результаты YOLO для кадра
    frame : np.ndarray
        Предобработанный кадр
    encoder : function
        Функция для извлечения признаков DeepSORT
    class_names : list
        Список имен классов
    conf_threshold : float
        Порог уверенности
    
    Returns:
    --------
    tuple : (frame_idx, bboxes, confidences, class_ids, names, features)
        Результаты обработки кадра
    """
    bboxes = []
    confidences = []
    class_ids = []
    
    # Извлекаем детекции из результатов YOLO
    try:
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes_data = results.boxes.data
            
            if boxes_data is not None:
                try:
                    if hasattr(boxes_data, 'numel'):
                        has_detections = boxes_data.numel() > 0
                    elif hasattr(boxes_data, '__len__'):
                        has_detections = len(boxes_data) > 0
                    else:
                        has_detections = True
                except:
                    has_detections = False
                
                if has_detections:
                    try:
                        if hasattr(boxes_data, 'cpu'):
                            boxes_array = boxes_data.cpu().numpy()
                        elif hasattr(boxes_data, 'numpy'):
                            boxes_array = boxes_data.numpy()
                        elif hasattr(boxes_data, 'tolist'):
                            boxes_list = boxes_data.tolist()
                            boxes_array = np.array(boxes_list)
                        else:
                            boxes_array = np.array(boxes_data)
                    except:
                        try:
                            boxes_array = np.array(boxes_data)
                        except:
                            boxes_array = None
                    
                    if boxes_array is not None:
                        if len(boxes_array.shape) == 2 and boxes_array.shape[1] >= 6:
                            for data in boxes_array:
                                x1, y1, x2, y2, confidence, class_id = data[0], data[1], data[2], data[3], data[4], data[5]
                                x = int(x1)
                                y = int(y1)
                                w = int(x2) - int(x1)
                                h = int(y2) - int(y1)
                                class_id = int(class_id)

                                if confidence > conf_threshold:
                                    bboxes.append([x, y, w, h])
                                    confidences.append(float(confidence))
                                    class_ids.append(class_id)
    except:
        pass
    
    # Получаем имена классов
    names = [class_names[class_id] for class_id in class_ids] if class_ids else []
    
    # Извлекаем признаки для DeepSORT
    features = encoder(frame, bboxes) if bboxes else []
    
    return (frame_idx, bboxes, confidences, class_ids, names, features)


def extract_detections_parallel(yolo_results, preprocessed_frames, encoder, class_names, conf_threshold=None, num_workers=None):
    """
    Параллельное извлечение детекций и признаков из результатов YOLO.
    
    Parameters:
    -----------
    yolo_results : list
        Список результатов YOLO
    preprocessed_frames : list
        Список предобработанных кадров
    encoder : function
        Функция для извлечения признаков DeepSORT
    class_names : list
        Список имен классов
    conf_threshold : float
        Порог уверенности
    num_workers : int
        Количество потоков для обработки
    
    Returns:
    --------
    detection_data : list
        Список кортежей (bboxes, confidences, class_ids, names, features) для каждого кадра
    """
    if conf_threshold is None:
        conf_threshold = config.CONF_THRESHOLD
    
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    
    # Проверяем соответствие количества результатов и кадров
    if len(yolo_results) != len(preprocessed_frames):
        print(f"Предупреждение: количество результатов YOLO ({len(yolo_results)}) не совпадает с количеством кадров ({len(preprocessed_frames)})")
        min_len = min(len(yolo_results), len(preprocessed_frames))
        yolo_results = yolo_results[:min_len]
        preprocessed_frames = preprocessed_frames[:min_len]
    
    detection_results = [None] * len(yolo_results)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Отправляем все задачи на обработку
        future_to_idx = {}
        for idx, (results, frame) in enumerate(zip(yolo_results, preprocessed_frames)):
            future = executor.submit(
                extract_detections_and_features,
                idx, results, frame, encoder, class_names, conf_threshold
            )
            future_to_idx[future] = idx
        
        # Собираем результаты в правильном порядке
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                detection_results[idx] = future.result()
                completed += 1
                if completed % 100 == 0 or completed == len(yolo_results):
                    print(f"  Обработано {completed} / {len(yolo_results)} кадров")
            except Exception as e:
                print(f"Ошибка при обработке кадра {idx}: {e}")
                # Создаем пустые данные для этого кадра
                detection_results[idx] = (idx, [], [], [], [], [])
    
    # Сортируем по индексу кадра для правильного порядка
    detection_results.sort(key=lambda x: x[0])
    detection_data = [result[1:] for result in detection_results]  # Убираем frame_idx
    
    return detection_data

