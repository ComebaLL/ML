__author__ = "Kuvykin Nikita"

from model_unit import predict
from module_download import load_model
import numpy as np

# Загружаем модель
try:
    model = load_model()
    print("Модель загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None

features1 = [-1.371047,	-1.033614,	-0.315653,	0.607639]
features2 = [1.027208,	0.770518,	0.688083,	-0.053609]
features3 = [-0.1996862017274523,	0.4174857898871989,	1.9131565574388292,	0.34618465051932246]
features4 = [-1.1554793563131376,	-0.8604470595817343,	0.16250335545793504,	-0.8040226132834724]
features5 = [0.28208759452617493,	0.42088342652514804,	0.8532918772325484,	-0.2938139669205567]
features6 = [-0.6036199760572349,	1.3203227067759158,	0.8273602627900478,	0.8846356875619851]
features7 = [1.1650070019404162,	2.1944579377316202,	-0.8235373178534059,	-0.14825603746176508]
features8 = [1.2074320030700758,	-0.5117090917911202,	-0.9189932063136543,	-0.9034080380090946]
features9 = [-0.7735219111503354,	0.5089491275981249,	0.04887501541454854,	1.7238711341434834]
features10 = [0.43395668318228364,	-0.38426038004559426,	0.477260237725489,	0.29118207178718003]

x1_pred = np.array([features1])
x2_pred = np.array([features2])
x3_pred = np.array([features3])
x4_pred = np.array([features4])
x5_pred = np.array([features5])
x6_pred = np.array([features6])
x7_pred = np.array([features7])
x8_pred = np.array([features8])
x9_pred = np.array([features9])
x10_pred = np.array([features10])

print(f"Предсказание 1 {model.predict(x1_pred)}")
print(f"Предсказание 2 {model.predict(x2_pred)}")
print(f"Предсказание 3 {model.predict(x3_pred)}")
print(f"Предсказание 4 {model.predict(x4_pred)}")
print(f"Предсказание 5 {model.predict(x5_pred)}")
print(f"Предсказание 6 {model.predict(x6_pred)}")
print(f"Предсказание 7 {model.predict(x7_pred)}")
print(f"Предсказание 8 {model.predict(x8_pred)}")
print(f"Предсказание 9 {model.predict(x9_pred)}")
print(f"Предсказание 10 {model.predict(x10_pred)}")

