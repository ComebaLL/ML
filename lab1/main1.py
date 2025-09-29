__author__ = "Kyvukin N.D"

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import matplotlib.pyplot as plt
import seaborn
import pandas as pd


# создание синтетического датасета для задачи регресии,
# 200 обхектов, 4 признака, noise - стандартное отклонение добавляемого шума,
# random_state - кол-во различных итерация
# n_infomative - кол-во признаков влияющих на целевую переменную
# X - признак
# Y - целевая переменная
# n_featuer - кол-во признаков 

X, Y = make_regression(200, n_features= 4, n_informative= 2, noise= 30, random_state=826456)

X += 100
Y += 300

df = pd.DataFrame(X, columns=['X1', 'X2', 'X3', 'X4'], )
df['Y'] = Y
df = df.round(4)
df.to_csv('reg_data.csv', index=False)


#data = pd.DataFrame({'X1': X[:,0], 'y' :Y})
plt.figure( figsize=(5,3))
plt.title("Диаграмма рассеивания")
seaborn.scatterplot( data = df, x = "X1", y = "y")
plt.show()

print (X.shape)

lin_reg = LinearRegression()    # модель линейной регресии
lin_reg.fit(X,Y)                # обучение модели

print( f"b1 {lin_reg.coef_}")            # коэффициент линейной регресии: b1 для x1,x2,x3,4

print( f"b0 {lin_reg.intercept_}")       # коэффициент линейной регресии: b0 свободный член


y_pred = lin_reg.predict(X)     # предсказание

# метрики качества
print( f"{mean_absolute_error( Y, y_pred)=:.2f}")       # средний модуль ощибки
print( f"{mean_squared_error( Y, y_pred)=:.2f}")        # средний квадрат ошибки
print( f"{r2_score( Y, y_pred)=:.2f}")                  # коэффициент детерминации



