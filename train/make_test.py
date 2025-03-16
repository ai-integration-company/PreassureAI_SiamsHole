import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
from scipy.interpolate import interp1d


def create_and_save_graphics(time, pressure, devpressure, subfolder_path, img_size=(256, 256, 3)):
    # Создаем белый фон изображения
    background = np.ones(img_size, dtype=np.uint8) * 255
    scatter_img = background.copy()
    plot_img = background.copy()
    os.makedirs(subfolder_path, exist_ok=True)
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


def process_sample(filename):
    curr_time_series = None
    with open("validation 1/validation 1/" + filename, "r") as f:
        curr_time_series = [tuple(map(float, line.replace("\n", "").split("\t"))) for line in f]

    time = np.array([triplet[0] for triplet in curr_time_series])
    pressure = np.array([triplet[1] for triplet in curr_time_series])
    derivative = np.array([triplet[2] for triplet in curr_time_series])

    if len(time) > 1:
        time = time - (min(time) - 1e-6)

        time = np.log(time)
        pressure = np.log(pressure)
        devpressure = np.log(derivative)

        time_interp = np.linspace(time.min(), time.max(), 10)

        # Интерполяция pressure и devpressure
        interp_pressure = interp1d(time, pressure, kind="linear", fill_value="extrapolate")
        interp_devpressure = interp1d(time, devpressure, kind="linear", fill_value="extrapolate")

        # Получение интерполированных значений
        pressure_interp = interp_pressure(time_interp)
        devpressure_interp = interp_devpressure(time_interp)

        create_and_save_graphics(time, pressure, devpressure, save_dir + filename)

        np_labels = np.array(
            [time[0], time[-1], pressure[0], devpressure[0]]
            + list(pressure_interp)
            + list(devpressure_interp)
        )
        np.save(save_dir + filename + "/labels.npy", np_labels)
    else:
        time = []
        pressure = []
        devpressure = []

        create_and_save_graphics(time, pressure, devpressure, save_dir + filename)
        np_labels = np.array([0, 0, 0, 0] + 20 * [0])
        np.save(save_dir + filename + "/labels.npy", np_labels)

    os.makedirs(save_dir + filename, exist_ok=True)


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


if __name__ == "__main__":

    # путь для валидации
    val_dir = r"C:\Users\vasil\research\siam_1_case\validation 1\validation 1"
    save_dir = "img_val_new_24labelsss/"
    os.makedirs(save_dir, exist_ok=True)

    image_folders = [f for f in os.listdir(val_dir)]

    num_threads = min(20, os.cpu_count())
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(
            tqdm(
                executor.map(process_sample, image_folders),
                total=len(image_folders),
                desc="Processing",
            )
        )
