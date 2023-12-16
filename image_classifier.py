import keras
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from os.path import isdir
from keras.layers import Dense, Flatten, Rescaling
import seaborn as sn
import matplotlib.pyplot as plt
import random


(train_dataset, test_dataset) = keras.utils.image_dataset_from_directory( # загружаем датасет
    "data-img", # из папки data-img
    label_mode="categorical", 
    image_size=(28, 28), # размер картинок 28 х 28
    validation_split=0.2, # отделяем 20% в тестовую выборку
    subset="both",
    seed=420,
    shuffle=True, # перемешиваем картинки
    color_mode="grayscale" # загружаем картинки в черно-белом формате
)

if not isdir("trained_model"):
    model = keras.Sequential([
        # выпрямляем массив из 28 x 28 в 28*28 = 784
        Flatten(input_shape=(28, 28, 1)), 
        # превращаем элементы массива из 0-255 в 0-1
        Rescaling(scale=1./255), 
        # полносвязный слой из 128 нейронов с функцией активации swish - x * sigmoid(x)
        Dense(128, activation="swish"), 
        # полносвязный слой из 128 нейронов с функцией активации swish - x * sigmoid(x)
        Dense(128, activation="swish"), 
        # полносвязный слой из 64 нейронов с функцией активации swish - x * sigmoid(x)
        Dense(64, activation="swish"), 
        # полносвязный слой из 5 нейронов с функцией активации softmax - 
        # преобразует вектор в распределение вероятностей с суммой 1
        Dense(5, activation="softmax"), 
    ])

    model.compile( # настраиваем обучение модели
        optimizer='adam', # используем алгоритм Адам для оптимизации
        loss='categorical_crossentropy', # используем кросс-энтропию как функцию ошибки для оптимизатора
        metrics=['accuracy']
    )
    model.fit(train_dataset, epochs=20, batch_size=20) # тренируем модель
    model.save("trained_model")
else:
    model = keras.models.load_model("trained_model")


x_test = np.concatenate([x for x, y in test_dataset], axis=0)

# предсказываем классы для тестовой выборки
y_pred = model.predict(test_dataset)
y_pred = np.argmax(y_pred, axis=1)
    
# собираем правильные классы из тестовой выборки
y_test = np.concatenate([y for x, y in test_dataset], axis=0)
y_test = np.argmax(y_test, axis=1)
    
# подсчитываем матрицу ошибок и отчет о точности классификации
matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(report)
sn.heatmap(matrix, annot=True, xticklabels=test_dataset.class_names,
            yticklabels=test_dataset.class_names, fmt=".2f")
plt.savefig("cm_model.png")

correct_indices = np.where(y_test == y_pred)[0]
incorrect_indices = np.where(y_test != y_pred)[0]

# Создайте подплоты для 1x5 изображений
fig, axes = plt.subplots(1, 5, figsize=(15, 15))

# Выведите 5 случайных правильных изображений
for i in range(5):
    ax = axes[i % 5]
    index = random.choice(correct_indices)
    image_data = x_test[index]  # Получите изображение из тестовых данных
    
    # Предсказание метки для изображения с помощью модели k-NN
    predicted_label = np.argmax(model.predict(np.array([image_data]))[0])
    actual_label = y_test[index]  # Получите реальную метку

    actual_label = test_dataset.class_names[actual_label]
    predicted_label = test_dataset.class_names[predicted_label]
    # Отобразите изображение и реальную метку
    ax.imshow(image_data, cmap='gray')
    ax.set_title(f"Actual Label: {actual_label}\nPredicted Label: {predicted_label}")
    ax.axis('off')  # Отключите оси координат


fig, axes = plt.subplots(1, 5, figsize=(15, 15))

# Выведите 5 случайных неправильных изображений
for i in range(5):
    ax = axes[i % 5]
    index = random.choice(incorrect_indices)
    image_data = x_test[index]  # Получите изображение из тестовых данных

    # Предсказание метки для изображения с помощью модели k-NN
    predicted_label = np.argmax(model.predict(np.array([image_data]))[0])
    actual_label = y_test[index]  # Получите реальную метку

    actual_label = test_dataset.class_names[actual_label]
    predicted_label = test_dataset.class_names[predicted_label]
    # Отобразите изображение
    ax.imshow(image_data, cmap='gray')
    ax.set_title(f"Actual Label: {actual_label}\nPredicted Label: {predicted_label}")
    ax.axis('off')  # Отключите оси координат

# Показать изображения в окне
plt.show()