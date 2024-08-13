import numpy as np
import pandas as pd
import seaborn as sb
import os
import fastai
from fastai.tabular.all import *

# Загрузка датасета
!pip install kaggle
!pip install opendatasets
import opendatasets as od

od.download("https://www.kaggle.com/datasets/uciml/iris")

# Чтение данных
train = pd.read_csv("/content/iris/Iris.csv")
train.info()

# Предобработка данных
train = train.drop('Id', axis=1)

# Определение категориальных и континуальных колонок
cat_names = ['Species']
cont_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Определение зависимой переменной
dep_var = 'Species'

# Создание DataLoaders
dls = TabularDataLoaders.from_df(train, y_names=dep_var, cat_names=cat_names, cont_names=cont_names, procs=[Categorify, FillMissing, Normalize])

# Создание модели
learn = tabular_learner(dls, metrics=accuracy)

# Обучение модели
learn.fit_one_cycle(5)

# Оценка модели
learn.show_results()
