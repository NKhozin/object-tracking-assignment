# Object Tracking Home assignment

### Задача

- Реализовать методы tracker_soft и tracker_strong в скрипте fastapi_server.py
- Придумать и реализовать метод для оценки качества разработанных трекеров
- Провести эксперименты с различными значениями tracks_amount, random_range, bb_skip_percent

### Описание трекеров
tracker_soft - венгерский алгоритм
tracker_strong - DeepSort
### Описание метрик
- Точность = сумма наибольших длин одинаковых треков для каждого cb_id / общее число треков
- Процент смены треков = кол-во смен идентификаторов при трекинге / общее число треков
- Средняя длина неизменяющейся последовательности = сумма длин всех неизменяющихся последовательностей треков / количество уникальных треков
### Эксперименты

|    | tracker   |   tracks_amount |   random_range |   bb_skip_percent |   accuracy |   change_percentage |   average_seq_length |
|---:|:----------|----------------:|---------------:|------------------:|-----------:|--------------------:|---------------------:|
|  0 | soft      |              10 |              1 |                10 |  0.287425  |            49.6914  |              1.95322 |
|  9 | soft      |              10 |              1 |                10 |  0.691617  |            20.0617  |              4.45333 |
|  1 | soft      |              10 |             10 |                 0 |  0.819672  |             2.37288 |             17.9412  |
|  2 | soft      |              10 |             10 |                25 |  0.124113  |            75.7353  |              1.30556 |
|  3 | soft      |              10 |             10 |                50 |  0.11399   |            87.4317  |              1.13529 |
|  4 | soft      |              10 |             20 |                25 |  0.146825  |            74.3802  |              1.32632 |
|  5 | soft      |              15 |             10 |                25 |  0.118863  |            81.1828  |              1.22082 |
|  6 | soft      |              20 |             10 |                25 |  0.0967153 |            84.8485  |              1.17094 |
|  8 | soft      |               5 |             10 |                25 |  0.256     |            56.6667  |              1.71233 |
| 13 | strong    |              10 |              1 |                10 |  0.706587  |            18.5185  |              4.77143 |
| 10 | strong    |              10 |             10 |                 0 |  0.711475  |            17.6271  |              4.91935 |
| 11 | strong    |              10 |             10 |                25 |  0.537879  |            25.5906  |              3.52    |
| 12 | strong    |              10 |             10 |                50 |  0.296296  |            48.6034  |              1.94845 |
| 14 | strong    |              10 |             20 |                25 |  0.370518  |            44.8133  |              2.12712 |
| 15 | strong    |              15 |             10 |                25 |  0.484474  |            28.6517  |              3.17094 |
| 16 | strong    |              20 |             10 |                25 |  0.438632  |            33.543   |              2.76111 |
| 18 | strong    |               5 |             10 |                25 |  0.416     |            27.5     |              3.28947 |


### Выводы
- При равных значениях параметров метрики tracker_strong выше чем tracker_soft
- С увеличением количества объектов значения метрик для обоих трекеров падают
- Увеличение ложного смещения рамки также негативно влияет на метрики обоих трекеров
- Метрики значительно растут, если вероятность не обнаружения объекта 0. Повышение вероятности не обнаружения объекта сильнее влияет на tracker_soft
