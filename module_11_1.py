import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup



# Код генерирует данные о продажах за неделю и отображает их на графике.
# Сначала генерирует случайные значения продаж для каждого дня недели,
# а затем строит столбчатую диаграмму.

days = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
sales = np.random.randint(100, 500, size = len(days))


plt.figure(figsize = (10, 5))
plt.bar(days, sales, color = 'blue')
plt.xlabel('День недели')
plt.ylabel('Продажи')
plt.title('Продажи за неделю')
plt.ylim(0, max(sales) + 100)


for i, v in enumerate(sales):
    plt.text(i, v + 10, str(v), ha = 'center')

plt.show()

# Визуализация графика

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [2, 1, 3, 4])
plt.show()

# Строим многорядные столбчатые диаграммы для анализа данных

index = np.arange(5)
data = {'series1':[1, 2, 3, 4, 5],
        'series2':[3, 2, 5, 4, 5],
        'series3':[4, 5, 3, 3, 3]}
df = pd.DataFrame(data)
df.plot(kind = 'bar')
plt.show()

# Рисуем 3D-куб

axes = [5, 5, 5]
data = np.ones(axes, dtype=np.bool)
alpha = 0.9
colors = np.empty(axes + [4], dtype=np.float32)
colors[:] = [1, 0, 0, alpha]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(data, facecolors=colors)
plt.show()

# Создание массива:
a = np.array([[1, 2, 3], [4, 5, 6]])

# Произведение всех элементов массива
prod_all = a.prod()
print(prod_all)

# Произведение элементов вдоль оси 0
prod_axis0 = a.prod(axis=0)
print(prod_axis0)

# Произведение элементов вдоль оси 1
prod_axis1 = a.prod(axis=1)
print(prod_axis1)



# Извлечем заголовки новостей с главной страницы журнала Xakep.ru.

response = requests.get('https://xakep.ru')
page = response.text

soup = BeautifulSoup(page, 'html.parser')

headings = map(lambda e: e.text, soup.select('h3.entry-title a span'))
for h in headings:
    print(h)

# Определяем ip адрес своего компьютера

req = requests.get('https://ifconfig.me')
print(f'Мой публичный IP-адрес: {req.text}')

# Прогноз погоды в Кирове на завтра

BASE_URL = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 58.5966,
    "longitude": 49.6601,
    "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum",
    "timezone": "Europe/Kirov"
}
response = requests.get(BASE_URL, params=params)
if response.status_code == 200:
    data = response.json()

    tomorrow_temp_min = data['daily']['temperature_2m_min'][1]
    tomorrow_temp_max = data['daily']['temperature_2m_max'][1]
    tomorrow_precipitation = data['daily']['precipitation_sum'][1]

    print(f"Прогноз погоды в Кирове на завтра:")
    print(f"Минимальная температура: {tomorrow_temp_min}°C")
    print(f"Максимальная температура: {tomorrow_temp_max}°C")
    print(f"Ожидаемое количество осадков: {tomorrow_precipitation} мм")
else:
    print(f"Ошибка {response.status_code}: {response.text}")

