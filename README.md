# Нейросеть U-Net для сегментации голосовой щели

model.py - структура модели, аугментация данных встроена в модель, слои аугментации вызываются только при обучении.

data.py - функции для предобработки данных.

main.ipynb - Jupyter ноутбук, с кодом для обучения.

main.py - вариант main.ipynb

segment_video.ipynb - используется для теста модели на целом видео, экспортирует видео-маску.
