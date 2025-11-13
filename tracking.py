"""
Модуль для отслеживания объектов с использованием DeepSORT.
"""

from collections import deque
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet
import config


class LineCounter:
    """Класс для подсчета пересечений линий."""
    
    def __init__(self, start_line, end_line, direction='down', max_track_id=1000, max_points=32):
        """
        Инициализация счетчика линий.
        
        Parameters:
        -----------
        start_line : tuple
            Начальная точка линии (x, y)
        end_line : tuple
            Конечная точка линии (x, y)
        direction : str
            Направление пересечения: 'down' (сверху вниз) или 'up' (снизу вверх)
        max_track_id : int
            Максимальный ID трека
        max_points : int
            Максимальное количество точек для отслеживания траектории
        """
        self.start_line = start_line
        self.end_line = end_line
        self.direction = direction
        self.counter = 0
        self.points = [deque(maxlen=max_points) for _ in range(max_track_id)]
    
    def check_crossing(self, track_id, center_x, center_y):
        """
        Проверяет пересечение линии объектом.
        
        Parameters:
        -----------
        track_id : int
            ID трека
        center_x : int
            X координата центра объекта
        center_y : int
            Y координата центра объекта
        
        Returns:
        --------
        bool
            True если произошло пересечение, False иначе
        """
        # Добавляем текущую точку
        self.points[track_id].append((center_x, center_y))
        
        # Проверяем пересечение
        if len(self.points[track_id]) < 2:
            return False
        
        # Получаем первую точку (самую старую)
        last_point_x = self.points[track_id][0][0]
        last_point_y = self.points[track_id][0][1]
        
        # Проверяем, что x координата в пределах линии
        if not (self.start_line[0] < center_x < self.end_line[0]):
            return False
        
        # Для линии A: движение сверху вниз
        # center_y > start_line_A[1] > last_point_y
        if self.direction == 'down':
            if center_y > self.start_line[1] > last_point_y:
                self.counter += 1
                self.points[track_id].clear()
                return True
        # Для линии B: движение снизу вверх
        # center_y < start_line_B[1] and last_point_y > start_line_B[1]
        elif self.direction == 'up':
            if center_y < self.start_line[1] and last_point_y > self.start_line[1]:
                self.counter += 1
                self.points[track_id].clear()
                return True
        
        return False
    
    def get_count(self):
        """Возвращает количество пересечений."""
        return self.counter


def create_tracker(max_cosine_distance=None, nn_budget=None):
    """
    Создает трекер DeepSORT.
    
    Parameters:
    -----------
    max_cosine_distance : float
        Максимальное косинусное расстояние для сопоставления
    nn_budget : int
        Бюджет для ближайших соседей
    
    Returns:
    --------
    tracker : Tracker
        Объект трекера DeepSORT
    encoder : function
        Функция для извлечения признаков
    """
    if max_cosine_distance is None:
        max_cosine_distance = config.MAX_COSINE_DISTANCE
    if nn_budget is None:
        nn_budget = config.NN_BUDGET
    
    # Инициализация DeepSORT
    model_filename = config.DEEPSORT_MODEL_PATH
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    return tracker, encoder


def update_tracker(tracker, detections):
    """
    Обновляет трекер с новыми детекциями.
    
    Parameters:
    -----------
    tracker : Tracker
        Объект трекера DeepSORT
    detections : list
        Список объектов Detection
    
    Returns:
    --------
    tracks : list
        Список активных треков
    """
    tracker.predict()
    tracker.update(detections)
    return tracker.tracks


def create_detections(bboxes, confidences, names, features):
    """
    Создает список объектов Detection для DeepSORT.
    
    Parameters:
    -----------
    bboxes : list
        Список ограничивающих рамок [x, y, w, h]
    confidences : list
        Список уверенностей
    names : list
        Список имен классов
    features : list
        Список признаков для каждого объекта
    
    Returns:
    --------
    detections : list
        Список объектов Detection
    """
    dets = []
    for bbox, conf, class_name, feature in zip(bboxes, confidences, names, features):
        dets.append(Detection(bbox, conf, class_name, feature))
    return dets

