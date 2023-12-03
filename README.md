# ML-homeworks-2023

Что было сделано:
  1. Предобработка числовых данных
       1. Заполнение пропусков медианными значениями
       2. Стандартизация значений
  2. Предобработка категориальных значений
       1. Преобразование признаков к удобному формату
       2. One-hot кодирование
  3. Выбор модели регрессии, подбор гиперпараметров
  4. Визуализация и анализ корреляций признаков
  5. Добавление новых признаков на основе текущих
  6. Построение pipeline
  7. Разработка сервиса на базе FastApi

С какими результатами:
  1. Лучше всего себя показала модель Ridge с логарифмированием таргета. На тестовой выборке R^2 = 0.88
     
Что дало наибольший буст в качестве:
  1. Добавление новых признаков и логарифмизация таргета
     
Что сделать не вышло и почему:
  1. Не предусмотрена ситуация, когда в тестовой выборке присутствует значение категориального признака, которого нет в трейне
  2. Создано не мало pkl-файлов. Скорее всего, это не очень оптимальное решение.

Ниже представлены результаты работы сервиса:

Для одного json:
<img width="1370" alt="Снимок экрана 2023-12-03 в 18 57 34" src="https://github.com/VladaTrus/ML-homework-1/assets/47668051/126b8adb-177b-4f8c-a845-8df2f6cdc259">

Для csv файла:
<img width="1370" alt="Снимок экрана 2023-12-03 в 18 58 12" src="https://github.com/VladaTrus/ML-homework-1/assets/47668051/424e34fb-9cfa-4f62-bf09-34231f7d326a">
<img width="1291" alt="Снимок экрана 2023-12-03 в 18 59 18" src="https://github.com/VladaTrus/ML-homework-1/assets/47668051/6989a2a6-0541-41ff-9fd2-3c5bcfdc396c">

