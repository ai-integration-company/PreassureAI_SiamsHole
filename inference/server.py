import os

import numpy as np

import torch
from torch import nn
from torchvision import models, transforms
import cv2
from scipy.interpolate import interp1d
from PIL import Image


bin_columns = [
    "Некачественное ГДИС",
    "Влияние ствола скважины",
    "Радиальный режим",
    "Линейный режим",
    "Билинейный режим",
    "Сферический режим",
    "Граница постоянного давления",
    "Граница непроницаемый разлом",
]
num_columns = [
    "Влияние ствола скважины_details",
    "Радиальный режим_details",
    "Линейный режим_details",
    "Билинейный режим_details",
    "Сферический режим_details",
    "Граница постоянного давления_details",
    "Граница непроницаемый разлом_details",
]


# Определение модели
class MultiTaskResNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.base = models.resnet101(pretrained=pretrained)

        # Преобразуем первый слой под 2 канала
        orig_conv = self.base.conv1
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias,
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = orig_conv.weight[:, :3, :, :]
            new_conv.weight[:, 3:, :, :] = orig_conv.weight[:, :3, :, :]
        self.base.conv1 = new_conv

        # Финальный FC: 15 выходов (8 – бинарные, 7 – регрессия)
        num_feats = self.base.fc.in_features
        self.base.fc = nn.Identity()

        self.binary = nn.Sequential(
            nn.Linear(num_feats, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 8),
        )

        self.resgresion = nn.Sequential(
            nn.Linear(num_feats + 24, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 7),
        )

    def forward(self, x, start_values=None):
        out = self.base(x)

        logits_8 = self.binary(out)
        if start_values is None:
            regr_7 = torch.sigmoid(
                self.resgresion(torch.cat((out, torch.zeros(x.size()[0], 24)), axis=1))
            )
        else:
            regr_7 = torch.sigmoid(self.resgresion(torch.cat((out, start_values), axis=1)))
        return logits_8, regr_7


RANGES = [
    (-0.5, 5.5),  # Влияние ствола скважины_details
    (-0.3, 4.1),  # Радиальный режим_details
    (-0.7, 2.2),  # Линейный режим_details
    (-0.4, 4.0),  # Билинейный режим_details
    (-0.3, 3.0),  # Сферический режим_details
    (2.4, 495.0),  # Граница постоянного давления_details
    (4.0, 555.0),  # Граница непроницаемый разлом_details
]


# Функция нормализации
def normalize_labels(num_labels):
    num_labels_scaled = num_labels.clone()  # Создаём копию
    for i in range(7):
        min_val, max_val = RANGES[i]
        mask = num_labels[:, i] != -1000  # Только для валидных значений
        num_labels_scaled[mask, i] = (num_labels[mask, i] - min_val) / (max_val - min_val)
    return num_labels_scaled.clip(0, 1)  # Обрезаем значения до [0,1]


# Функция обратного преобразования
def denormalize_labels(pred_labels):
    pred_labels_orig = pred_labels.clone()
    for i in range(7):
        min_val, max_val = RANGES[i]
        pred_labels_orig[:, i] = pred_labels[:, i] * (max_val - min_val) + min_val
    return pred_labels_orig


def create_and_save_graphics(time, pressure, devpressure, img_size=(256, 256, 3)):
    # Создаем белый фон изображения
    background = np.ones(img_size, dtype=np.uint8) * 255
    scatter_img = background.copy()
    plot_img = background.copy()
    # Если временной ряд пустой, сохраняем пустые изображения и выходим
    if len(time) == 0:
        # Сохраняем пустые изображения как PNG
        return plot_img, scatter_img

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

    return plot_img, scatter_img


def process_sample(curr_time_series):

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

        plot_img, scatter_img = create_and_save_graphics(time, pressure, devpressure)

        np_labels = np.array(
            [time[0], time[-1], pressure[0], devpressure[0]]
            + list(pressure_interp)
            + list(devpressure_interp)
        )
    else:
        time = []
        pressure = []
        devpressure = []

        plot_img, scatter_img = create_and_save_graphics(time, pressure, devpressure)
        np_labels = np.array([0, 0, 0, 0] + 20 * [0])

    return plot_img, scatter_img, np_labels


transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = MultiTaskResNet()

    state_dict = torch.load(
        r"trainingv1\models\resnet18_300v5_0.9294251187380995_0.9989427171440238.pth",
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model = model.to(device)

    model.eval()

    # путь до файла на котором будет инференс
    timeseries_path = (
        r"C:\Users\vasil\research\siam_1_case\data\data\0a0a0068-0bc0-4e62-bb7a-4ef8a50e590d"
    )

    curr_time_series = None
    with open(timeseries_path, "r") as f:
        curr_time_series = [tuple(map(float, line.replace("\n", "").split("\t"))) for line in f]

    plot_img, scatter_img, np_labels = process_sample(curr_time_series)

    with torch.no_grad():

        plot_img = transform(Image.fromarray(plot_img))

        scatter_img = transform(Image.fromarray(scatter_img))

        res_img = torch.cat([plot_img, scatter_img], dim=0).to(device).float()

        start_values = torch.tensor(np_labels)

        start_values = start_values.unsqueeze(0).to(device).float()
        logits_8, regr_7 = model(res_img.unsqueeze(0), start_values)

        pred_binary = torch.sigmoid(logits_8).cpu().numpy().flatten()

        pred_regression = regr_7.cpu().numpy()

        pred_regression = denormalize_labels(torch.tensor(pred_regression)).cpu().numpy().flatten()

        pred_binary = (pred_binary >= 0.5).astype(int)

    print("Бинарные признаки ", pred_binary)
    print("Численные признаки ", pred_regression)
