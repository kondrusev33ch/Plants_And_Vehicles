import matplotlib.pyplot as plt
import pandas as pd
import json
from tabulate import tabulate
from PIL import Image


def show_image(file: str):
    with Image.open(file) as img:
        print('Path:', file)
        print('Size:', img.size)
        plt.imshow(img)
        plt.show()


# --------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Get data
    # ========
    with open('data.json', 'r', encoding='utf8') as f:
        annotations = json.load(f)

    # Prepare data
    # ============
    print(annotations.keys())
    # => dict_keys(['initial_bundle', 'test_bundle'])

    initial_b = annotations['initial_bundle']
    print(initial_b[0].keys())
    # => dict_keys(['id', 'type', 'category', 'subcategory', 'tags', 'file'])
    print(len(initial_b))
    # => 1558

    test_b = annotations['test_bundle']
    print(test_b[0].keys())
    # => dict_keys(['id', 'type', 'category', 'subcategory', 'tags', 'file'])
    print(len(test_b))
    # => 1545

    # Перенесём все данные в pandas dataframe для удобства
    initial_df = pd.read_json(json.dumps(initial_b))
    test_df = pd.read_json(json.dumps(test_b))

    # Analyse data
    # ============
    # Для начала давайте взглянем на таблицы по отдельности
    print(tabulate(initial_df.head(), headers='keys'))
    # ... таблица занимает много места
    print(tabulate(test_df.head(), headers='keys'))
    # ... таблица занимает много места

    # Теперь давайте объединим эти таблицы
    data_df_full = pd.concat([initial_df, test_df], ignore_index=True)
    print(data_df_full.shape)
    # => (3103, 6)

    assert len(data_df_full['id'].unique()) == data_df_full.shape[0], 'id duplications found!'

    # Проверим в каждом ли ряду есть путь к изображению
    print(data_df_full['file'].isna().sum())
    # => 0

    # Наша задача построить и обучить модель которая будет выявлять машины и растения
    # Поэтому нас интересуют все категории связанные с растениями и машинами

    # У нас 2 missing values
    print(data_df_full.loc[pd.isna(data_df_full.loc[:, 'category']), :].index)
    # => Int64Index([2058, 2425], dtype='int64')
    print(data_df_full.loc[[2058, 2425], 'category'])
    # 2058    None
    # 2425    None
    # Name: category, dtype: object

    # Давайте взглянем вдруг у них есть 'subcategory', 'tags' или 'file'
    print(tabulate(data_df_full.loc[[2058, 2425], ['subcategory', 'tags', 'file']], headers='keys'))
    #       subcategory    tags                    file
    # ----  -------------  ----------------------  ---------------------------------------------
    # 2058                 ['helix', 'houdini']    data/3586ff8d-98c4-4480-a644-63e43e120f55.png
    # 2425                 ['houdini', 'normals']  data/a240223c-da46-45ff-bfe5-b14887289860.png

    # Эти изображения креветки и спирали нам не интересны, можно спокойно дропать
    data_df = data_df_full.drop([2058, 2425])
    print(data_df.shape)
    # => (3101, 6)

    # Теперь посмотрим на все категории
    categories = {}
    for d in data_df.loc[:, 'category']:
        if d['name'] in categories:
            categories[d['name']] += 1
        else:
            categories[d['name']] = 1

    print(dict(sorted(categories.items(), key=lambda item: item[1], reverse=True)))
    # {'Light Textures': 427, 'Vehicles': 357, 'Plants': 265, 'Explosions': 142, 'Other': 125,
    # 'Studios': 123, 'Human': 114, 'Outdoor': 110, 'Architecture': 108, 'Buildings': 105, 'Interior': 86,
    # 'Abstract': 66, 'Windows': 63, 'Space': 58, 'Debris': 56, 'Sparks': 55, 'Industrial': 51,
    # 'Particles': 50, 'Mapping': 50, 'People': 48, 'Kitbash': 47, 'Splashes': 46, 'cyber': 41,
    # 'Sky': 37, 'Science': 33, 'Stages': 32, 'Dust': 31, 'Smoke': 28, 'Indoor': 25, 'Simulation': 20,
    # 'Noises': 19, 'Typography': 19, 'Light': 18, 'Objects': 17, 'Shapes': 16, 'SciFi': 16,
    # 'Installations': 15, 'Lines': 13, 'Stock': 12, 'Electronics': 11, 'Background': 11,
    # 'Toys and games': 10, 'Flame': 10, 'Gradients': 9, 'Nature': 9, 'Exterior': 8, 'Digital': 8,
    # 'Real People': 8, 'Glass': 7, 'Creatures': 5, 'Dome': 5, 'Walls': 5, 'Stone': 5, 'Clouds': 3,
    # 'Presentations': 3, 'Signs & Symbols': 3, 'Grids & Dots': 3, 'Impulse': 3, 'Map': 3,
    # 'LED Facades': 2, 'Animals': 2, 'Data Flows': 2, 'Grids': 2, 'Identity': 2, 'Video': 2,
    # 'Squares': 2, 'Performances': 1, 'Stones': 1, 'Pointers': 1, 'Flares': 1, 'Grunge': 1,
    # 'Special': 1, 'Metals': 1, 'Dirt': 1, 'Wipe Mark': 1, 'Patterns': 1, 'Presale Videos': 1,
    # 'Terrain': 1, 'Art': 1, 'Music': 1}

    # Нам интересны только "Plants" и "Vehicles"
    plants_ids = []
    vehicles_ids = []

    for _, row in data_df.iterrows():
        if row['category']['name'] == 'Plants':
            plants_ids.append(row['id'])
        if row['category']['name'] == 'Vehicles':
            vehicles_ids.append(row['id'])

    plants_df = data_df.loc[data_df['id'].isin(plants_ids), :]
    vehicles_df = data_df.loc[data_df['id'].isin(vehicles_ids), :]

    # Небольшая проверка
    print(plants_df.shape)
    # => (265, 6)
    print(vehicles_df.shape)
    # => (357, 6)

    # Теперь посмотрим на изображения
    for file in plants_df['file'].sample(3):  # 3 для примера
        show_image(file)
    
    for file in vehicles_df['file'].sample(3):
        show_image(file)

    # Разные размеры, разные форматы и изображения не подходящие под категории машины и растения

    # Изображения растений, которые я счёл неподходящими для тренировки модели
    pla_outliers = ['data/f93d1da6-5d02-4971-a6f0-22baf1e3b818.png']  # большое кол-во мелких элементов

    # Изображения машин, которые я счёл неподходящими для тренировки модели
    veh_outliers = ['data/906a18ff-4486-45eb-8c4f-e67258d6c094.png',  # здание
                    'data/92f4b8f1-bcb6-41d0-aa50-1544128f2ec9.png',  # терминал
                    'data/775c4b67-98b1-44e9-9b11-65972063bd68.png',  # двигатель
                    'data/cbe420c2-64ea-4495-8627-62a043ce3330.png',  # двигатель
                    'data/21a08826-8b51-44b4-8d82-e34e3f338f53.png',  # кресла
                    'data/fe053c60-0c40-4c55-b20b-ab82789165b0.png',  # вертолет да Винчи
                    'data/df9b6873-c9cc-4d7f-82a8-a3fe5579559c.png',  # авто-подвеска
                    'data/dff87634-28f9-4d7e-b3fd-dc9b97db4b58.png',  # махолёт да Винчи
                    'data/42edef17-d8f5-4a92-adb8-40e07012d841.png',  # капсула, ракета
                    'data/ce7bbe01-4d7c-4a07-ab88-963231a299fc.png',  # электро-двигатель
                    'data/f4281d30-0ea1-4f49-91a6-dcd0823060a4.png',  # депо, вагоны, локомотив
                    'data/686edad9-9e4a-415d-9c1b-a81c782e5874.png',  # грузовой контейнер
                    'data/4150ccab-997d-437c-9b6b-2c2b92f4ce0c.png',  # колесо
                    'data/80c7f1ba-1409-4342-bbda-6d36f9409457.png',  # футуристическое авто
                    'data/ea99f7c3-d4cb-4afb-af47-81320d2538a5.png',  # наливник
                    'data/60e8e520-d381-497b-b37f-ba06963bcddd.png',  # двигатель
                    'data/927a27b0-0a41-456f-b0a5-b3e42e89d8c6.png',  # конструкции
                    'data/59ea3db7-be9c-4fcb-8bdc-1a4015429d3d.png',  # вертолёт
                    'data/3b8414cb-7588-4087-9677-b87c47e79d49.png',  # станция депо
                    'data/4a6cafc7-387b-44ad-9525-5c48b9de569d.png',  # вертолёт
                    'data/f9ed39ab-7f4d-46e8-9a2a-86f07469e7c4.png',  # судно
                    'data/106b806c-9a24-4ddd-be35-df15d045c5c5.png',  # сет футуристических авто (to small)
                    'data/6e1654ce-92ab-488f-9ead-e88587db8823.png']  # двигатель с элементами охлаждения

    plants_df_rdy = plants_df[~(plants_df['file'].isin(pla_outliers))]
    vehicles_df_rdy = vehicles_df[~(vehicles_df['file'].isin(veh_outliers))]
    print(plants_df_rdy.shape)
    # => (264, 6)
    print(vehicles_df_rdy.shape)
    # => (334, 6)

    # Присваеваем класс каждому изображению, сохраняем и можем приступать к заданию
    data_df_full['class'] = 0
    data_df_full.loc[data_df_full['id'].isin(plants_df_rdy['id']), 'class'] = 1
    data_df_full.loc[data_df_full['id'].isin(vehicles_df_rdy['id']), 'class'] = 2

    data_df_full.to_csv('data.csv', index=False)

    # Conclusion
    # ==========
    # Проведён анализ данных при помощи которого мы выделили группы с машинами и растениями
