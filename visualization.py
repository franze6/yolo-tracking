"""
Модуль для визуализации результатов детекции и отслеживания.
"""

import cv2
import numpy as np
import config


def load_class_names(classes_path=None):
    """
    Загружает имена классов из файла.
    
    Parameters:
    -----------
    classes_path : str
        Путь к файлу с именами классов
    
    Returns:
    --------
    class_names : list
        Список имен классов
    colors : np.ndarray
        Массив цветов для каждого класса
    """
    if classes_path is None:
        classes_path = config.CLASSES_PATH
    
    with open(classes_path, "r") as f:
        class_names = f.read().strip().split("\n")
    
    # Создаем список случайных цветов для каждого класса
    np.random.seed(42)  # для получения одинаковых цветов
    colors = np.random.randint(0, 255, size=(len(class_names), 3))
    
    return class_names, colors


def draw_counting_lines(frame, start_line_A, end_line_A, start_line_B, end_line_B, alpha=0.5):
    """
    Рисует линии подсчета на кадре.
    
    Parameters:
    -----------
    frame : np.ndarray
        Кадр для рисования
    start_line_A : tuple
        Начальная точка линии A
    end_line_A : tuple
        Конечная точка линии A
    start_line_B : tuple
        Начальная точка линии B
    end_line_B : tuple
        Конечная точка линии B
    alpha : float
        Прозрачность линий
    
    Returns:
    --------
    frame : np.ndarray
        Кадр с нарисованными линиями
    """
    overlay = frame.copy()
    
    # Рисуем линии
    cv2.line(frame, start_line_A, end_line_A, (0, 255, 0), 12)
    cv2.line(frame, start_line_B, end_line_B, (255, 0, 0), 12)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return frame


def draw_track(frame, track, class_names, colors, excluded_classes=None):
    """
    Рисует трек объекта на кадре.
    
    Parameters:
    -----------
    frame : np.ndarray
        Кадр для рисования
    track : Track
        Объект трека из DeepSORT
    class_names : list
        Список имен классов
    colors : np.ndarray
        Массив цветов для классов
    excluded_classes : list
        Список классов для исключения
    
    Returns:
    --------
    center : tuple or None
        Координаты центра объекта (x, y) или None если класс исключен
    """
    if excluded_classes is None:
        excluded_classes = config.EXCLUDED_CLASSES
    
    # Получаем ограничивающую рамку, ID трека и имя класса
    bbox = track.to_tlbr()
    track_id = track.track_id
    class_name = track.get_class()
    
    # Пропускаем исключенные классы
    if class_name in excluded_classes:
        return None
    
    # Преобразуем ограничивающую рамку в целые числа
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    # Получаем цвет для класса
    class_id = class_names.index(class_name)
    color = colors[class_id]
    B, G, R = int(color[0]), int(color[1]), int(color[2])
    
    # Рисуем ограничивающую рамку
    text = str(track_id) + " - " + class_name
    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 3)
    cv2.rectangle(frame, (x1 - 1, y1 - 20),
                  (x1 + len(text) * 12, y1), (B, G, R), -1)
    cv2.putText(frame, text, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Вычисляем центр объекта
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # Рисуем центр объекта
    cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)
    
    return (center_x, center_y)


def draw_counters(frame, counter_A, counter_B):
    """
    Рисует счетчики пересечений на кадре.
    
    Parameters:
    -----------
    frame : np.ndarray
        Кадр для рисования
    counter_A : int
        Счетчик для линии A
    counter_B : int
        Счетчик для линии B
    """
    cv2.putText(frame, "Line A:", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Line B:", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"{counter_A}", (180, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"{counter_B}", (180, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

