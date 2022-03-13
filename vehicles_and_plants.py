"""
Задание:
    Используя python и этот датасет постройте и обучите модель, которая среди предложенных изображений
    выявляет машины и растения, и выводит на экран список файлов в каждой категории с указанием
    вероятности совпадения.
"""

import albumentations as A
import pandas as pd
import numpy as np
import random
import os

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models
from helpers import plot_training_curves, plot_batch_images
from tabulate import tabulate
from tqdm import tqdm
from PIL import Image

IMG_WIDTH = 200
IMG_HEIGHT = 200
BATCH_SIZE = 8
N_EPOCHS = 300  # с учётом использования EarlyStopping callback
MODEL_NAME = 'ResNet50'


def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_image(img_path):
    # Я выбрал способ конвертации png изображения в jpg через библиотеку pillow, но просто
    # сконвертировать в jpg не получится, вокруг объекта появляется шум, который помешает нашей
    # модели обучаться. Ответ был найден на stackoverflow.
    with Image.open(img_path) as img:
        if img.mode != 'RGB':
            bg = Image.new('RGB', img.size, (255, 255, 255))  # создаём белый бэкграунд
            bg.paste(img, img)  # накладываем изображение на бэкграунд
            img = bg

        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)

    return np.asarray(img)


# Data generator
# =====================================================================================================
class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset.copy().sample(frac=1.0).reset_index(drop=True)  # shuffle dataset
        self.n_labels = self.dataset['class'].nunique()  # get number of classes
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset) // BATCH_SIZE

    def __get_output(self, label):
        return keras.utils.to_categorical(label, num_classes=self.n_labels)

    def __get_input(self, img_path):
        img = get_image(img_path)

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img.astype(np.float32) / 255.0

    def __getitem__(self, index):
        batches = self.dataset[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]

        images = np.asarray([self.__get_input(img_path) for img_path in batches['file']])
        labels = np.asarray([self.__get_output(label) for label in batches['class']])

        return images, labels

    def on_epoch_end(self):
        self.dataset = self.dataset.sample(frac=1.0).reset_index(drop=True)


# Augmentation
# =====================================================================================================
def get_transforms():
    return A.Compose([A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                                    val_shift_limit=0.2, p=0.9),
                               A.RandomBrightnessContrast(brightness_limit=0.2,
                                                          contrast_limit=0.2, p=0.9)], p=0.9),
                      A.HorizontalFlip(p=0.5),
                      A.CoarseDropout(max_holes=8, max_height=10, max_width=10, fill_value=0, p=0.5),
                      A.Rotate(limit=20, p=0.5)],
                     p=1.0)


# Model
# =====================================================================================================
def prepare_model_for_transfer_learning(base_model, num_classes):
    model = models.Sequential()

    model.add(base_model)
    model.add(layers.Flatten())

    model.add(layers.Dense(1024, activation='relu', input_dim=512))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def get_resnet_model(input_shape, num_classes):
    base_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights='imagenet', input_shape=in_shape)

    model = prepare_model_for_transfer_learning(base_model, num_classes)
    model.build((None, *input_shape))
    return model


# Training
# =====================================================================================================
def run(model, t_generator, v_generator, callbacks, name):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    t_history = model.fit(t_generator,
                          validation_data=v_generator,
                          epochs=N_EPOCHS,
                          callbacks=callbacks)
    model.save_weights(f'{name}_weights.ckpt')

    return t_history


# -----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    seed_everything()

    # Data preparing
    # ===============
    # Взглянем на соотношение количетва изображений в каждом из классов
    data_df = pd.read_csv('data.csv')
    print(data_df.shape)
    # => (3103, 7)

    cls_proportion = data_df.groupby('class')['class'].count()
    print(cls_proportion)
    # class
    # 0    2505   -> Other
    # 1     264   -> Plant
    # 2     334   -> Vehicle

    # И сразу создадим map с понятной расшифровкой классов
    classes_map = {0: 'Other', 1: 'Plant', 2: 'Vehicle'}

    # Так как у нас сильный перевес в сторону Other, нам нужно корректно создать валидационный сет
    valid_size = int(cls_proportion.min() * 0.2)  # 20% от класса с минимальным количеством элементов
    valid_df = pd.DataFrame()
    for cls in data_df['class'].unique():
        df = data_df[data_df['class'] == cls]
        valid_df = valid_df.append(df.sample(valid_size))

    # Таким образов валидационный сет содержит равное количество элементов из кажодого класса
    print(valid_df.groupby('class')['class'].count())
    # class
    # 0    52
    # 1    52
    # 2    52

    valid_df = valid_df.sort_index()
    print(valid_df.shape)
    # => (156, 7)

    # Создадим тренировочный сет
    train_df = data_df.drop(valid_df.index)

    # Создадим тестовый датасет
    test_df = pd.DataFrame()
    for cls in data_df['class'].unique():
        df = train_df[train_df['class'] == cls]
        test_df = test_df.append(df.sample(20))  # 20 элементов из каждого класса я счёл оптимальным вариантом
    print(test_df.shape)
    # => (60, 7)

    # Обновляем тренировочный сет
    train_df = train_df.drop(test_df.index)  # исключаем тестовый датасет из тренировочного
    print(train_df.shape)
    # => (2887, 7)

    # Будем использовать кастомный data generator
    train_generator = DataGenerator(train_df, get_transforms())
    valid_generator = DataGenerator(valid_df)

    # Прежде чем перейти к обучению, давайте взглянем на то, какие изображения у нас получается генерировать
    plot_batch_images(train_generator, classes_map)

    # Model creation and training
    # ===========================
    # EarlyStopping callback
    callback = [keras.callbacks.EarlyStopping(patience=5,
                                              monitor='val_accuracy',
                                              restore_best_weights=True)]
    in_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    # Создаём модель
    resnet = get_resnet_model(in_shape, len(cls_proportion))

    # Обучаем модель
    resnet_history = run(resnet, train_generator, valid_generator, callback, MODEL_NAME)
    plot_training_curves(resnet_history)

    # Загружаем лучшие веса
    resnet.load_weights(f'{MODEL_NAME}_weights.ckpt')

    # Results
    # =======
    # Генерируем таблицу с результатами с использованием тестого датасета
    test_df['class'] = test_df['class'].map(lambda x: classes_map[x])
    test_df = test_df.sample(frac=1.0).reset_index(drop=True)
    prediction_percents = []
    prediction_classes = []
    for path in tqdm(test_df['file']):
        image = get_image(path) / 255.0

        prediction = resnet.predict(np.expand_dims(image, axis=0))
        prediction_percents.append(prediction[0].max())
        prediction_classes.append(classes_map[prediction[0].argmax()])

    test_df['pred_class'] = prediction_classes
    test_df['pred_percent'] = prediction_percents

    results = test_df[['file', 'class', 'pred_class', 'pred_percent']]
    print(tabulate(results, headers='keys'))  # в задании стоит задача вывести на экран
    #     file                                           class    pred_class      pred_percent
    # --  ---------------------------------------------  -------  ------------  --------------
    #  0  data/f25a9e9f-e188-4f5e-9d9f-a054d82ac769.jpg  Other    Other               1
    #  1  data/d8ee3a51-289c-4964-ace0-98bf8167da21.png  Vehicle  Vehicle             1
    #  2  data/fd055611-d13c-4287-b3ef-1c984bc6f70b.png  Vehicle  Vehicle             0.723876
    #  3  data/6026efc5-23cc-454f-8269-b843e0e4d02b.png  Plant    Plant               0.925067
    #  4  data/6ba0a180-ad77-4a4a-82f4-d921c18729b4.png  Plant    Plant               0.999999
    #  5  data/391e477a-6c7b-488f-be25-e38a69d8446e.png  Other    Other               0.957531
    #  .  ...                                            ...      ...                 ...
    # 55  data/c8fdf1b6-7cef-4ac1-b269-7cb291a5bd6a.png  Vehicle  Vehicle             0.994519
    # 56  data/39172786-367a-4b3f-b536-504c8f10ded5.png  Vehicle  Vehicle             0.999925
    # 57  data/639f2e3d-2752-4f03-8e20-8ed07e6ea7bc.png  Other    Other               0.999995
    # 58  data/56daf54d-3ae6-4674-91db-3565b6e496fa.png  Vehicle  Vehicle             0.999968
    # 59  data/58eed508-0789-4e76-938a-9318f749c49c.png  Plant    Plant               0.996081

    results.to_csv('final_results.csv', index=False)

    diff = (results['class'] == results['pred_class'])
    print('Accuracy:', diff.sum() / len(diff))
    # Accuracy: 1.0

    # Conclusion
    # ==========
    # В заключение хочу сказать, что ResNet50 показывает себя хорошо в таких задачах, конечно мы можем
    # построить свою модель с нуля, но результаты не будут также хороши при тех же затратах времени и
    # мощностей
    # С увеличением сложности задачи, я бы добавил ансамбль из нескольких моделей, но в
    # данном задании это будет лишним
