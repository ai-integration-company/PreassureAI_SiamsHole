import os
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image

import re


import random

from torch.utils.data import Dataset
from scipy.interpolate import interp1d


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        time_serires_path=r"C:\Users\vasil\research\siam_1_case\data\data",
        data_path=r"C:\Users\vasil\research\siam_1_case\markup_train.csv",
        type_labels="all",  # Может быть 'all', 'binary', 'skips'
    ):
        self.data = pd.read_csv(data_path)
        self.data = self.data.fillna(-1000)
        self.time_serires_path = time_serires_path
        self.type_labels = type_labels

        # Заменим все строки, с номерами пользователей и департаментов на номера
        self.data["department_name"] = self.data["department_name"].apply(
            lambda x: int(re.sub(r"\D", "", x))
        )
        self.data["user_name"] = self.data["user_name"].apply(lambda x: int(re.sub(r"\D", "", x)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]

        filename = row["file_name"]

        curr_time_series = None
        with open(os.path.join(self.time_serires_path, filename), "r") as f:
            curr_time_series = [tuple(map(float, line.replace("\n", "").split("\t"))) for line in f]

        if self.type_labels == "skips":
            # Отбираем только те признаки которые существовали, то есть не были NaN
            discribe_series = {
                column: row[column]
                for column in self.data.iloc[:, 1:].columns
                if row[column] != -1000
            }
        elif self.type_labels == "binary":
            discribe_series = {column: row[column] for column in self.data.iloc[:, 3:11].columns}
        else:
            # Отбираем все признаки
            discribe_series = {column: row[column] for column in self.data.columns}

        return curr_time_series, discribe_series


def create_and_save_graphics(time, pressure, devpressure, subfolder_path, img_size=(256, 256, 3)):
    # Создаем белый фон изображения
    background = np.ones(img_size, dtype=np.uint8) * 255
    scatter_img = background.copy()
    plot_img = background.copy()

    # Если временной ряд пустой, сохраняем пустые изображения и выходим
    if len(time) == 0:
        # Сохраняем пустые изображения как PNG
        Image.fromarray(plot_img).save(os.path.join(subfolder_path, "plot.png"))
        Image.fromarray(scatter_img).save(os.path.join(subfolder_path, "scatter.png"))
        return

    # Находим минимальные и максимальные значения для времени, давления и производной
    time_min = min(time)
    time_max = max(time)
    pressure_min = min(pressure)
    pressure_max = max(pressure)
    devpressure_min = min(devpressure)
    devpressure_max = max(devpressure)

    # Для нормализации по оси y берем общий минимум и максимум для давления и производной
    max_y = max(devpressure_max, pressure_max)
    min_y = min(devpressure_min, pressure_min)

    # Строим сетку: вертикальные линии с шагом 0.5
    grid_x = np.array([x / 2 for x in range(int(2 * time_min), int(2 * time_max) + 1)])
    # Горизонтальные линии для оси y, тоже с шагом 0.5
    grid_y = np.array([x / 2 for x in range(int(2 * min_y), int(2 * max_y) + 1)])

    # Нормализация данных для времени и значений:
    # Защита от деления на ноль
    if time_max - time_min == 0:
        norm_time = np.zeros(len(time), dtype=np.int32)
        grid_x_norm = np.zeros_like(grid_x, dtype=np.int32)
    else:
        norm_time = (
            (np.array(time) - time_min) / (time_max - time_min) * (img_size[1] - 10)
        ).astype(np.int32)
        grid_x_norm = ((grid_x - time_min) / (time_max - time_min) * (img_size[1] - 10)).astype(
            np.int32
        )

    if max_y - min_y == 0:
        norm_pressure = np.zeros(len(pressure), dtype=np.int32)
        norm_devpressure = np.zeros(len(devpressure), dtype=np.int32)
        grid_y_norm = np.zeros_like(grid_y, dtype=np.int32)
    else:
        norm_pressure = (
            (np.array(pressure) - min_y) / (max_y - min_y) * (img_size[0] - 10)
        ).astype(np.int32)
        norm_pressure = img_size[0] - norm_pressure
        norm_devpressure = (
            (np.array(devpressure) - min_y) / (max_y - min_y) * (img_size[0] - 10)
        ).astype(np.int32)
        norm_devpressure = img_size[0] - norm_devpressure
        grid_y_norm = ((grid_y - min_y) / (max_y - min_y) * (img_size[0] - 10)).astype(np.int32)
        grid_y_norm = img_size[0] - grid_y_norm

    # Рисуем вертикальные линии на обоих изображениях (сетка)
    for x in grid_x_norm:
        cv2.line(
            scatter_img, (int(x), 0), (int(x), img_size[0]), color=(128, 128, 128), thickness=1
        )
        cv2.line(plot_img, (int(x), 0), (int(x), img_size[0]), color=(128, 128, 128), thickness=1)

    # Рисуем горизонтальные линии на обоих изображениях (сетка)
    for y in grid_y_norm:
        cv2.line(
            scatter_img, (0, int(y)), (img_size[1], int(y)), color=(128, 128, 128), thickness=1
        )
        cv2.line(plot_img, (0, int(y)), (img_size[1], int(y)), color=(128, 128, 128), thickness=1)

    # Рисуем scatter-график: точки для давления (синий) и производной (красный)
    for i in range(len(time)):
        cv2.circle(scatter_img, (int(norm_time[i]), int(norm_pressure[i])), 1, (255, 0, 0), -1)
        cv2.circle(scatter_img, (int(norm_time[i]), int(norm_devpressure[i])), 1, (0, 0, 255), -1)

    # Рисуем plot-график: соединяем точки линиями для давления и производной
    for i in range(len(time) - 1):
        cv2.line(
            plot_img,
            (int(norm_time[i]), int(norm_pressure[i])),
            (int(norm_time[i + 1]), int(norm_pressure[i + 1])),
            color=(255, 0, 0),
            thickness=1,
        )
        cv2.line(
            plot_img,
            (int(norm_time[i]), int(norm_devpressure[i])),
            (int(norm_time[i + 1]), int(norm_devpressure[i + 1])),
            color=(0, 0, 255),
            thickness=1,
        )

    # Преобразуем numpy массивы в изображения с помощью PIL и сохраняем как PNG
    plot_pil = Image.fromarray(plot_img)
    scatter_pil = Image.fromarray(scatter_img)

    plot_pil.save(os.path.join(subfolder_path, "plot.png"))
    scatter_pil.save(os.path.join(subfolder_path, "scatter.png"))

    return


def apply_augmentations(
    time, press, deriv, numeric_labels, time_shift_range=0.07, slope_scale_range=0.035
):
    """

    time = log(original time)
    press = log(original press)

    press = log(original press)

    time_shift_range - сдвиг вдоль оси x

    slope_scale_range - маштабированние данных (почти все таргеты определяются по углу наклона графика, поэтому лучше брать маленькое значение)
    """

    shift_factor = random.uniform(-time_shift_range, time_shift_range)

    median_t = np.median(time)

    actual_shift = shift_factor * median_t

    time += actual_shift

    time_dependent_features = [
        "Граница постоянного давления_details",
        "Граница непроницаемый разлом_details",
    ]

    for key in time_dependent_features:

        if key in numeric_labels and numeric_labels[key] != -1000:

            numeric_labels[key] += actual_shift

    slope_factor = random.uniform(-slope_scale_range, slope_scale_range)

    press *= 1.0 + slope_factor

    deriv *= 1.0 + slope_factor

    slope_dependent_features = [
        "Влияние ствола скважины_details",
        "Радиальный режим_details",
        "Линейный режим_details",
        "Билинейный режим_details",
        "Сферический режим_details",
    ]

    for key in slope_dependent_features:

        if key in numeric_labels and numeric_labels[key] != -1000:

            numeric_labels[key] *= 1.0 + slope_factor

    return time, press, deriv, numeric_labels


def save_images(
    dataset1=None,
    dataset2=None,
    aug=apply_augmentations,
    count_best_aug=8,
    count_all_aug=3,
    root_dir="images",
):
    """
    Функция создает папку root_dir в которой сохраняются графики и лейблы
    dataset1 - Датасет с качественными данными, которые будут count_best_aug раз аугментированны
    dataset2 - Все данные среди которых есть и менее качественные представители, аугментировать будем count_all_aug раз
    !!!! в итоговом наборе данных будет count_best_aug + count_all_aug примеров из dataset1 так они есть и в dataset2
    aug - фнукция аугментации
    root_dir - имя папки в которую сохраняются данные
    """
    binary_features = [
        "Некачественное ГДИС",
        "Влияние ствола скважины",
        "Радиальный режим",
        "Линейный режим",
        "Билинейный режим",
        "Сферический режим",
        "Граница постоянного давления",
        "Граница непроницаемый разлом",
    ]
    float_features = [
        "Влияние ствола скважины_details",
        "Радиальный режим_details",
        "Линейный режим_details",
        "Билинейный режим_details",
        "Сферический режим_details",
        "Граница постоянного давления_details",
        "Граница непроницаемый разлом_details",
    ]

    if dataset1 is None:
        dataset1 = TimeSeriesDataset(
            data_path=r"C:\Users\vasil\research\siam_1_case\hq_markup_train.csv"
        )
    if dataset2 is None:
        dataset2 = TimeSeriesDataset()

    # Создаем корневую папку
    os.makedirs(root_dir, exist_ok=True)

    number_sample = (
        1  # Так как будет разное кол-во аугментаций, будем отдельно контролировать номер примера
    )

    # Сначала создаем картинки из лучших данных
    for idx in tqdm(
        range(len(dataset1)), desc="create images for best dataset"
    ):  # Не ебу почему, но если делать x, y in  dataset1 он за границы датасета выходит
        label = dataset1[idx][1]
        time_series = np.array(dataset1[idx][0])

        time = np.log(time_series[:, 0])
        pressure = np.log(time_series[:, 1])
        devpressure = np.log(time_series[:, 2])

        time_interp = np.linspace(time.min(), time.max(), 10)

        # Интерполяция pressure и devpressure
        interp_pressure = interp1d(time, pressure, kind="linear", fill_value="extrapolate")
        interp_devpressure = interp1d(time, devpressure, kind="linear", fill_value="extrapolate")

        # Получение интерполированных значений
        pressure_interp = interp_pressure(time_interp)
        devpressure_interp = interp_devpressure(time_interp)

        # Создадим папку для sample
        subfolder_name = f"sample{number_sample}"
        subfolder_path = os.path.join(root_dir, subfolder_name)
        if os.path.exists(subfolder_path):
            number_sample += 1 + count_best_aug
            continue  # при желании можно выйти или делать что-то ещё

        os.makedirs(subfolder_path, exist_ok=True)

        create_and_save_graphics(time, pressure, devpressure, subfolder_path)

        np_labels = np.array(
            [label.get(key, -1000) for key in binary_features]
            + [label.get(key, -1000) for key in float_features]
            + [time[0], time[-1], pressure[0], devpressure[0]]
            + list(pressure_interp)
            + list(devpressure_interp)
        )
        np.save(os.path.join(subfolder_path, "labels.npy"), np_labels)
        number_sample += 1

        # Аугментируем count_best_aug раз
        for _ in range(count_best_aug):
            new_time, new_pressure, new_devpressure, new_label = apply_augmentations(
                time,
                pressure,
                devpressure,
                label,
                time_shift_range=0.07,
                slope_scale_range=0.035,
            )

            new_time_interp = np.linspace(new_time.min(), new_time.max(), 10)

            # Интерполяция pressure и devpressure
            new_interp_pressure = interp1d(
                new_time, new_pressure, kind="linear", fill_value="extrapolate"
            )
            new_interp_devpressure = interp1d(
                new_time, new_devpressure, kind="linear", fill_value="extrapolate"
            )

            # Получение интерполированных значений
            new_pressure_interp = new_interp_pressure(new_time_interp)
            new_devpressure_interp = new_interp_devpressure(new_time_interp)

            # Создадим папку для sample
            subfolder_name = f"sample{number_sample}"
            subfolder_path = os.path.join(root_dir, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)

            create_and_save_graphics(new_time, new_pressure, new_devpressure, subfolder_path)

            np_labels = np.array(
                [new_label.get(key, -1000) for key in binary_features]
                + [new_label.get(key, -1000) for key in float_features]
                + [new_time[0], new_time[-1], new_pressure[0], new_devpressure[0]]
                + list(new_pressure_interp)
                + list(new_devpressure_interp)
            )
            np.save(os.path.join(subfolder_path, "labels.npy"), np_labels)
            number_sample += 1

    # Создаем картинки из всех данных
    for idx in tqdm(
        range(len(dataset2)), desc="create images for all dataset"
    ):  # Не ебу почему, но если делать x, y in  dataset1 он за границы датасета выходит
        label = dataset2[idx][1]
        time_series = np.array(dataset2[idx][0])

        # Создадим папку для sample
        subfolder_name = f"sample{number_sample}"
        subfolder_path = os.path.join(root_dir, subfolder_name)
        if os.path.exists(subfolder_path):
            number_sample += 1 + count_all_aug
            continue  # при желании можно выйти или делать что-то ещё

        os.makedirs(subfolder_path, exist_ok=True)

        if len(time_series) > 1:
            time = time_series[:, 0]
            time = time - (min(time) - 1e-6)

            time = np.log(time)
            pressure = np.log(time_series[:, 1])
            devpressure = np.log(time_series[:, 2])
        else:
            time = []
            pressure = []
            devpressure = []
            create_and_save_graphics(time, pressure, devpressure, subfolder_path)

            np_labels = np.array(
                [label.get(key, -1000) for key in binary_features]
                + [label.get(key, -1000) for key in float_features]
                + [0, 0, 0, 0]
                + 20 * [0]
            )
            np.save(os.path.join(subfolder_path, "labels.npy"), np_labels)
            number_sample += 1

            continue

        time_interp = np.linspace(time.min(), time.max(), 10)

        # Интерполяция pressure и devpressure
        interp_pressure = interp1d(time, pressure, kind="linear", fill_value="extrapolate")
        interp_devpressure = interp1d(time, devpressure, kind="linear", fill_value="extrapolate")

        # Получение интерполированных значений
        pressure_interp = interp_pressure(time_interp)
        devpressure_interp = interp_devpressure(time_interp)

        create_and_save_graphics(time, pressure, devpressure, subfolder_path)

        np_labels = np.array(
            [label.get(key, -1000) for key in binary_features]
            + [label.get(key, -1000) for key in float_features]
            + [time[0], time[-1], pressure[0], devpressure[0]]
            + list(pressure_interp)
            + list(devpressure_interp)
        )
        np.save(os.path.join(subfolder_path, "labels.npy"), np_labels)
        number_sample += 1

        # Аугментируем временные ряды count_all_aug раз
        for _ in range(count_all_aug):
            new_time, new_pressure, new_devpressure, new_label = apply_augmentations(
                time,
                pressure,
                devpressure,
                label,
                time_shift_range=0.07,
                slope_scale_range=0.035,
            )

            new_time_interp = np.linspace(new_time.min(), new_time.max(), 10)

            # Интерполяция pressure и devpressure
            new_interp_pressure = interp1d(
                new_time, new_pressure, kind="linear", fill_value="extrapolate"
            )
            new_interp_devpressure = interp1d(
                new_time, new_devpressure, kind="linear", fill_value="extrapolate"
            )

            # Получение интерполированных значений
            new_pressure_interp = new_interp_pressure(new_time_interp)
            new_devpressure_interp = new_interp_devpressure(new_time_interp)

            # Создадим папку для sample
            subfolder_name = f"sample{number_sample}"
            subfolder_path = os.path.join(root_dir, subfolder_name)
            os.makedirs(subfolder_path, exist_ok=True)

            create_and_save_graphics(new_time, new_pressure, new_devpressure, subfolder_path)

            np_labels = np.array(
                [new_label.get(key, -1000) for key in binary_features]
                + [new_label.get(key, -1000) for key in float_features]
                + [new_time[0], new_time[-1], new_pressure[0], new_devpressure[0]]
                + list(new_pressure_interp)
                + list(new_devpressure_interp)
            )
            np.save(os.path.join(subfolder_path, "labels.npy"), np_labels)
            number_sample += 1

        # Аугментируем временные ряд еще 3 раза, в случае если один из последних принаков присутствует
        # '''
        # Не будем делать так как не хватает памяти
        # '''
        if label[binary_features[-1]] == 1 or label[binary_features[-2]] == 1:
            for _ in range(3):
                new_time, new_pressure, new_devpressure, new_label = apply_augmentations(
                    time,
                    pressure,
                    devpressure,
                    label,
                    time_shift_range=0.07,
                    slope_scale_range=0.035,
                )

                new_time_interp = np.linspace(new_time.min(), new_time.max(), 10)

                # Интерполяция pressure и devpressure
                new_interp_pressure = interp1d(
                    new_time, new_pressure, kind="linear", fill_value="extrapolate"
                )
                new_interp_devpressure = interp1d(
                    new_time, new_devpressure, kind="linear", fill_value="extrapolate"
                )

                # Получение интерполированных значений
                new_pressure_interp = new_interp_pressure(new_time_interp)
                new_devpressure_interp = new_interp_devpressure(new_time_interp)

                # Создадим папку для sample
                subfolder_name = f"sample{number_sample}"
                subfolder_path = os.path.join(root_dir, subfolder_name)
                os.makedirs(subfolder_path, exist_ok=True)

                create_and_save_graphics(new_time, new_pressure, new_devpressure, subfolder_path)

                np_labels = np.array(
                    [new_label.get(key, -1000) for key in binary_features]
                    + [new_label.get(key, -1000) for key in float_features]
                    + [new_time[0], new_time[-1], new_pressure[0], new_devpressure[0]]
                    + list(new_pressure_interp)
                    + list(new_devpressure_interp)
                )
                np.save(os.path.join(subfolder_path, "labels.npy"), np_labels)
                number_sample += 1

    return


if __name__ == "__main__":

    # пропишите пути до данных и csv файла
    path_to_data = r"C:\Users\vasil\research\siam_1_case\data\data"
    path_to_train_csv = r"C:\Users\vasil\research\siam_1_case\markup_train.csv"
    path_hq_train_csv = r"C:\Users\vasil\research\siam_1_case\hq_markup_train.csv"
    root_dir = "images"

    dataset = TimeSeriesDataset(path_to_data, path_to_train_csv)
    hq_dataset = TimeSeriesDataset(path_to_data, path_hq_train_csv)

    save_images(hq_dataset, dataset, count_best_aug=10, count_all_aug=4, root_dir=root_dir)
