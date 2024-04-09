from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.optimize import linear_sum_assignment
from fastapi import FastAPI, WebSocket
from itertools import groupby
from PIL import Image
import numpy as np
import inspect
import asyncio
import pickle
import glob
import os

from tracks.track_11 import track_data, country_balls_amount
import tracks.track_11 as track_11

# Получение название файла, из которого происходит импорт для логирования
file_path = inspect.getfile(track_11)
file_name = os.path.splitext(os.path.basename(file_path))[0]

# Выбор типа трекера. Вынес сюда для быстрого изменения совместно с импортируемым файлом track_n
method_type = 'strong'

app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')

# Информация по параметрам импортируемых файлов для логирования
params_dict = {
    'track_5': '5_10_25',
    'track_6': '10_10_25',
    'track_7': '10_10_0',
    'track_8': '10_1_10',
    'track_9': '10_20_25',
    'track_10': '10_10_50',
    'track_11': '15_10_25',
    'track_12': '20_10_25',
}

# Получение параметров для текущего запуска в зависимости от файла, из которого был импорт
current_params = params_dict[file_name]

# Чтение файла с метрики по прошлым экспериментам
metric_dict = pickle.load(open("metric_dict.pickle", 'rb'))


def calculate_distance_bbox(bbox1, bbox2):
    #
    """
    Функция для вычисления расстояния между двумя bounding box
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    center1 = np.array([x1 + 0.5*w1, y1 + 0.5*h1])
    center2 = np.array([x2 + 0.5*w2, y2 + 0.5*h2])
    return np.linalg.norm(center1 - center2)


def calculate_distance_centers(center1, center2):
    """
    Функция для вычисления расстояния между двумя центрами bounding box
    """
    return np.linalg.norm(np.array(center1) - np.array(center2))


def find_closest_track_id(bboxes_tracks, bbox_center):
    """
    Функция поиска ближайшего трека
    """

    min_distance = float('inf')
    closest_track_id = None

    for track_id, track_center in bboxes_tracks.items():

        distance = calculate_distance_centers(track_center, bbox_center)

        if distance < min_distance:
            min_distance = distance
            closest_track_id = track_id

    return closest_track_id


def tracker_soft(el, next_track_id, prev_frame_objects):
    """
    Присваивает каждому объекту идентификатор (track_id) на основе венгерского алгоритма
    """
    curr_frame_objects = el['data']

    if not prev_frame_objects:

        for i, obj in enumerate(curr_frame_objects):
            obj['track_id'] = i
            next_track_id += 1

        return el, next_track_id

    cost_matrix = []

    for obj in curr_frame_objects:

        cost_row = []

        for prev_obj in prev_frame_objects:
            if obj['bounding_box'] and prev_obj['bounding_box']:
                cost_row.append(calculate_distance_bbox(obj['bounding_box'], prev_obj['bounding_box']))

        if cost_row:
            cost_matrix.append(cost_row)

    if cost_matrix:

        cost_matrix = np.array(cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for row, col in zip(row_ind, col_ind):
            curr_frame_objects[row]['track_id'] = prev_frame_objects[col]['track_id']

    for obj in curr_frame_objects:

        if not obj['bounding_box']:
            obj['track_id'] = None

        elif 'track_id' not in obj or obj['track_id'] is None:
            obj['track_id'] = next_track_id
            next_track_id += 1

    return el, next_track_id


def tracker_strong(tracker, el):
    """
    Присваивает каждому объекту идентификатор (track_id) на основе DeepSort
    """

    frame_id = el['frame_id']

    try:
        img_path = f'frames/{file_name}/{frame_id}.png'
        im_pil = Image.open(img_path)
        frame = np.array(im_pil.convert('RGB'))

        bboxes = []

        for idx, x in enumerate(el['data']):
            if x['bounding_box']:
                x1, y1, x2, y2 = x['bounding_box']
                w, h = x2 - x1, y2 - y1
                bboxes.append(([x1, y1, w, h], 1.0, 'cball'))

        tracks = tracker.update_tracks(bboxes, frame=frame)

        # Преобразование координат треков в формат центра и размеров для YOLO
        bboxes_tracks = {
            track.track_id: [(track.to_ltrb()[0] + track.to_ltrb()[2]) / 2, (track.to_ltrb()[1] + track.to_ltrb()[3]) / 2]
            for track in tracks
        }

    except Exception as e:
        print(e)
        bboxes_tracks = []

    for idx, x in enumerate(el['data']):

        if x['bounding_box'] and bboxes_tracks:

            bbox_center = [(x['bounding_box'][0] + x['bounding_box'][2]) / 2, (x['bounding_box'][1] + x['bounding_box'][3]) / 2]

            track_id = find_closest_track_id(bboxes_tracks, bbox_center)

            x['track_id'] = track_id

    return el


def calculate_metrics(metric):
    """
    Расчет метрик
    """

    # Очистка от None
    cleaned_metric = {key: [item for item in value if item is not None] for key, value in metric.items()}

    # Точность = сумма наибольших длин одинаковых треков для каждого cb_id / общее число треков
    total_length = sum(len(value) for value in cleaned_metric.values())
    longest_seqs_sum = sum(len(max((list(g) for k, g in groupby(value)), key=len)) for value in cleaned_metric.values())
    accuracy = longest_seqs_sum / total_length

    # Процент смены значения = кол-во смен идентификаторов при трекинге / общее число треков
    changes = sum(sum(1 for i in range(1, len(value)) if value[i] != value[i - 1]) for value in cleaned_metric.values())
    total_elements_minus_firsts = total_length - len(cleaned_metric)
    change_percentage = changes / total_elements_minus_firsts * 100

    # Средняя длина неизменяющейся последовательности
    all_seqs_lengths = [len(list(g)) for value in cleaned_metric.values() for k, g in groupby(value)]
    average_seq_length = sum(all_seqs_lengths) / len(all_seqs_lengths)

    return accuracy, change_percentage, average_seq_length


def update_stat(tracks_stat, el):
    """
    Обновление массивов для расчета итоговых метрик
    """

    print(el)
    for country_ball in el['data']:
       cb_id = country_ball['cb_id']
       if cb_id in tracks_stat:
            tracks_stat[cb_id].append(country_ball['track_id'])

    return tracks_stat


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    print('Accepting client connection...')
    await websocket.accept()

    global prev_frame_objects
    global next_track_id

    next_track_id = 0
    prev_frame_objects = None

    tracker = DeepSort(max_age=10, embedder='clip_ViT-B/16') #, max_iou_distance=0.8, max_cosine_distance=0.5,

    metric = {i: [] for i in range(country_balls_amount)}

    await websocket.send_text(str(country_balls))

    for el in track_data:

        await asyncio.sleep(0.1)

        if method_type == 'soft':
            el, next_track_id = tracker_soft(el, next_track_id, prev_frame_objects)
            prev_frame_objects = [obj for obj in el['data']]
        else:
            el = tracker_strong(tracker, el)

        metric = update_stat(metric, el)

        await websocket.send_json(el)

    # Расчет метрик
    accuracy, change_percentage, average_seq_length = calculate_metrics(metric)

    print(f"Точность: {accuracy}")
    print(f"Процент смены значения: {change_percentage}%")
    print(f"Средняя длина незименяющейся последовательности: {average_seq_length}")

    # Логирование метрик
    metric_dict[method_type+'_'+current_params] = [accuracy, change_percentage, average_seq_length]
    pickle.dump(metric_dict, open("metric_dict.pickle", "wb"))