"""
Главный файл для детекции и отслеживания объектов в видео.
"""

import cv2
import datetime
import config
import preprocessing
import detection
import tracking
import visualization

def load_video_frames(video_path):
    """
    Загружает все кадры из видео.
    
    Parameters:
    -----------
    video_path : str
        Путь к видео файлу
    
    Returns:
    --------
    frames : list
        Список кадров
    video_cap : cv2.VideoCapture
        Объект видеозахвата (для получения параметров)
    """
    video_cap = cv2.VideoCapture(video_path)
    frames = []
    
    print("Загрузка кадров из видео...")
    while True:
        ret, frame = video_cap.read()
        if not ret or frame is None:
            break
        frames.append(frame.copy())
    
    total_frames = len(frames)
    print(f"Загружено {total_frames} кадров")
    
    return frames, video_cap


def main():
    """Главная функция приложения."""
    start_all = datetime.datetime.now()

    print(cv2.cuda.getCudaEnabledDeviceCount())

    exit()
    
    # Загружаем кадры из видео
    frames, video_cap = load_video_frames(config.INPUT_VIDEO)
    
    # Получаем параметры видео для создания writer
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = config.DEFAULT_FPS
    
    # Закрываем видеозахват, так как все кадры загружены
    video_cap.release()
    
    # Загружаем модель YOLO
    print("Загрузка модели YOLO...")
    model = detection.load_model()
    
    # Инициализируем трекер DeepSORT
    print("Инициализация трекера DeepSORT...")
    tracker, encoder = tracking.create_tracker()
    
    # Загружаем имена классов и цвета
    class_names, colors = visualization.load_class_names()
    
    # Предобработка кадров
    print(f"Предобработка кадров (используется {config.NUM_WORKERS} процессов)...")
    preprocessed_frames = preprocessing.preprocess_frames_parallel(frames, config.NUM_WORKERS)
    print("Предобработка завершена")
    
    # Определяем размер кадров после предобработки (может измениться при upscaling)
    if len(preprocessed_frames) > 0:
        output_height, output_width = preprocessed_frames[0].shape[:2]
        if config.PREPROCESS_UPSCALE:
            print(f"Размер кадров после upscaling: {output_width}x{output_height}")
    else:
        output_width, output_height = frame_width, frame_height
    
    # Создаем writer для выходного видео с учетом возможного изменения размера
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(config.OUTPUT_VIDEO, fourcc, fps, (output_width, output_height))
    
    # Обработка YOLO батчами
    print(f"Обработка YOLO батчами размером {config.BATCH_SIZE}...")
    yolo_results = detection.process_frames_batch(model, preprocessed_frames, config.BATCH_SIZE, device=0)
    print("YOLO обработка завершена")
    print(f"Всего результатов YOLO: {len(yolo_results)}")
    
    # Параллельное извлечение детекций и признаков
    print(f"Параллельное извлечение детекций и признаков (используется {config.NUM_WORKERS} потоков)...")
    detection_data = detection.extract_detections_parallel(
        yolo_results, preprocessed_frames, encoder, class_names, config.CONF_THRESHOLD, config.NUM_WORKERS
    )
    print("Извлечение детекций и признаков завершено")
    
    # Инициализируем счетчики линий
    # Линия A: движение сверху вниз
    line_counter_A = tracking.LineCounter(
        config.START_LINE_A, config.END_LINE_A, 
        direction='down',
        max_track_id=config.MAX_TRACK_ID, 
        max_points=config.MAX_TRACK_POINTS
    )
    # Линия B: движение снизу вверх
    line_counter_B = tracking.LineCounter(
        config.START_LINE_B, config.END_LINE_B,
        direction='up',
        max_track_id=config.MAX_TRACK_ID,
        max_points=config.MAX_TRACK_POINTS
    )
    
    # Последовательная обработка с трекингом
    print("Обработка с трекингом...")
    frames_count = 0
    
    for frame_idx, (results, frame, (bboxes, confidences, class_ids, names, features)) in enumerate(
        zip(yolo_results, preprocessed_frames, detection_data)
    ):
        if frame is None:
            break
        
        frames_count += 1
        
        # Рисуем линии подсчета
        frame = visualization.draw_counting_lines(
            frame, 
            config.START_LINE_A, config.END_LINE_A,
            config.START_LINE_B, config.END_LINE_B
        )
        
        # Создаем детекции для DeepSORT
        dets = tracking.create_detections(bboxes, confidences, names, features)
        
        # Обновляем трекер
        tracks = tracking.update_tracker(tracker, dets)
        
        # Обрабатываем треки
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            # Рисуем трек и получаем центр объекта
            center = visualization.draw_track(frame, track, class_names, colors)
            
            if center is not None:
                center_x, center_y = center
                
                # Проверяем пересечения линий
                line_counter_A.check_crossing(track.track_id, center_x, center_y)
                line_counter_B.check_crossing(track.track_id, center_x, center_y)
        
        # Рисуем счетчики
        visualization.draw_counters(
            frame, 
            line_counter_A.get_count(), 
            line_counter_B.get_count()
        )
        
        # Записываем кадр
        writer.write(frame)
        
        # Прогресс
        if (frame_idx + 1) % 100 == 0 or (frame_idx + 1) == len(yolo_results):
            print(f"Обработано {frame_idx + 1} / {len(yolo_results)} кадров")
    
    end_all = datetime.datetime.now()
    
    # Выводим статистику
    fps = frames_count / (end_all - start_all).total_seconds()
    print(f"FPS: {fps:.2f}")
    print(f"Всего обработано кадров: {frames_count}")
    print(f"Пересечений линии A (Green): {line_counter_A.get_count()}")
    print(f"Пересечений линии B (Blue): {line_counter_B.get_count()}")
    
    # Освобождаем ресурсы
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

