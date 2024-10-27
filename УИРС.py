import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Функции для моделирования различных методов шифрования
def simulate_aes(data_size, key_length):
    energy_consumption = 0.1 * data_size * key_length / 128
    processing_time = 0.5 * data_size * key_length / 256
    resistance = "Высокая"
    return energy_consumption, processing_time, resistance

def simulate_rsa(data_size, key_length):
    energy_consumption = 0.5 * data_size * key_length / 2048
    processing_time = 2 * data_size * key_length / 2048
    resistance = "Средняя"
    return energy_consumption, processing_time, resistance

def simulate_qkd(data_size, distance):
    # Без использования оптимизации (статический расчет)
    energy_consumption = 0.3 * data_size * distance / 1000
    processing_time = 1.0 * data_size * distance / 1000
    resistance = "Очень высокая"
    return energy_consumption, processing_time, resistance

def simulate_qkd_optimized(data_size, distance):
    # Модель расчета с использованием МО
    def model_qkd(x, w):
        return np.dot(x, w)
    
    # Обучение модели с использованием функции потерь
    def loss_function(w, x, y):
        predictions = model_qkd(x, w)
        return np.mean(np.maximum(0, 1 - y * predictions))
    
    # Пример данных для обучения
    x_train = np.array([[data_size, distance]])
    y_train = np.array([1])  # Метка указывает на успешное обнаружение атаки (1 - атака обнаружена)
    
    # Оптимизация весов модели
    initial_weights = np.array([0.1, 0.1])
    result = minimize(loss_function, initial_weights, args=(x_train, y_train), method='BFGS')
    optimized_weights = result.x

    # Рассчитываем энергопотребление и время обработки с учетом оптимизации
    energy_consumption = 0.2 * data_size * distance / 1000 * (1 / np.dot(x_train, optimized_weights))
    processing_time = 0.8 * data_size * distance / 1000 * (1 / np.dot(x_train, optimized_weights))
    resistance = "Очень высокая"
    
    # Преобразуем результаты в числовой формат для корректного отображения
    energy_consumption = float(energy_consumption)
    processing_time = float(processing_time)
    
    # Возвращаем оптимизированные веса в виде строки для отображения в таблице
    weights_str = ", ".join([f"{w:.2f}" for w in optimized_weights])
    
    return energy_consumption, processing_time, resistance, weights_str

def run_simulation(data_size, key_length, distance):
    aes_result = simulate_aes(data_size, key_length)
    rsa_result = simulate_rsa(data_size, key_length)
    qkd_result = simulate_qkd(data_size, distance)
    qkd_optimized_result = simulate_qkd_optimized(data_size, distance)

    results = {
        "Метод": ["AES-256", "RSA", "QKD", "QKD (Оптимизированный)"],
        "Энергопотребление (ед.)": [aes_result[0], rsa_result[0], qkd_result[0], qkd_optimized_result[0]],
        "Время обработки (сек.)": [aes_result[1], rsa_result[1], qkd_result[1], qkd_optimized_result[1]],
        "Устойчивость": [aes_result[2], rsa_result[2], qkd_result[2], qkd_optimized_result[2]],
        "Оптимизированные веса (QKD)": ["-", "-", "-", qkd_optimized_result[3]]
    }

    return results

# Интерфейс приложения
st.title("Моделирование шифрования для спутников с QKD и его оптимизацией")
data_size = st.number_input("Размер данных (КБ):", min_value=1, value=100)
key_length = st.number_input("Длина ключа (бит):", min_value=1, value=256)
distance = st.number_input("Расстояние до наземной станции (км):", min_value=1, value=1000)

if st.button("Запустить моделирование"):
    results = run_simulation(data_size, key_length, distance)
    st.write("Результаты моделирования:")
    st.table(results)

    methods = results["Метод"]
    energy = [float(e) for e in results["Энергопотребление (ед.)"]]  # Убедимся, что все значения числовые
    time_processing = [float(t) for t in results["Время обработки (сек.)"]]  # Убедимся, что все значения числовые

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Методы')
    ax1.set_ylabel('Энергопотребление', color='tab:red')
    bars = ax1.bar(methods, energy, color='tab:red', alpha=0.6, label='Энергопотребление')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Время обработки', color='tab:blue')

    # Исправляем положение синих точек по центру столбцов
    for bar, time in zip(bars, time_processing):
        ax2.plot(bar.get_x() + bar.get_width() / 2, time, 'bo')
    
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    st.pyplot(fig)
