import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from scipy.stats import uniform
from scipy.optimize import curve_fit



chat_id = 627635260 # Ваш chat ID, не меняйте название переменной

def solution(x: np.ndarray) -> float:
    time_interval = 10  # время измерения скорости в секундах
    speeds = np.ndarray(x)  # преобразуем список в numpy array
    times = np.arange(1, len(speeds)+1) * time_interval  # время в секундах, в течение которого измерялась скорость

    # Определяем функцию, которая аппроксимирует зависимость скорости от времени
    def linear_function(t, a):
        return a * t

    # Используем метод наименьших квадратов для оценки коэффициента ускорения
    popt, pcov = curve_fit(linear_function, times, speeds)
    # Возвращаем значение коэффициента ускорения
    return popt[0] * 2


