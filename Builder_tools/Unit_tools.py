#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pathlib
import numpy as np
import dlib
import cv2
import mediapipe as mp
import tensorflow as tf
import keras
from keras_tuner import HyperModel
from keras.models import Sequential
from keras.layers import Dense,  Flatten, Dropout, BatchNormalization, Activation, Input
from Builder_tools.preprocess_func import preprocess_input_ResNet50, preprocess_input_VGG16
from sklearn.metrics import pairwise_distances
import time
import matplotlib.pyplot as plt
plt.style.use('dark_background')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


class MediaPiPe_detector:
    
    """
    MediaPiPe_detector - обьект детектирования лиц.
    
    def find_faces -> функция, которая на вход принимает изображение, конвертирует его в формат RGB, детектирует лица, 
                      получает координаты лица, преобразует прямоугольник с детектированным лицом в квадратную форму, без изменения изображения,
                      и на выходе возвращает массив координат 
    
    """
    
    def __init__(self):
        # Создание детектора из библиотеки MediaPipe
        self.detector = mp.solutions.face_detection 
    
    def find_faces(self, img):
        # Конвертируем изображение в формат RGB
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        
        # model_selection = 0, обнаружение лиц в пределах до 2-х метрах
        with self.detector.FaceDetection(model_selection=0, 
                                         min_detection_confidence=0.6) as face_detection:
            # Детектируем лица на изображении
            faces = face_detection.process(img)
            # Проверяем условие наличия лица на изображении
            if faces:
                coordinates_arr = np.ones((len(faces.detections),4), dtype='uint16')
                for e_num, face in enumerate(faces.detections):
                    bbox = face.location_data.relative_bounding_box
                    # Получаем значения координат расположения лица на изображении
                    mul_h = img.shape[0]
                    mul_w = img.shape[1]
                    x1 = int(abs(bbox.xmin) * mul_w)
                    y1 = int(abs(bbox.ymin) * mul_h)
                    h = int(abs(bbox.width) * mul_w)
                    w = int(abs(bbox.height) * mul_h)
                    x2 = x1 + w
                    y2 = y1 + h
                    
                    # Получаем значение высоты и ширины зоны детектирования лица
                    hight = y2 - y1
                    wight = x2 - x1
                    # Расчитываем макмиальную длину стороны прямоугольника
                    max_side = max(hight, wight)
                    # расчитываем значение для смещения координат по y
                    b_y = (hight - max_side) // 2
                    # расчитываем значение для смещения координат по x
                    b_x = (wight - max_side) // 2
                    # Вносим коррективы в значения по оси y
                    y1 += b_y
                    y2 += b_y
                    # Вносим коррективы в значения по оси x
                    x1 += b_x
                    x2 += b_x

                    # Сохраняю координаты детектированного лица с корректировкой расширения зоны лица
                    coordinates_arr[e_num][0] = max(int(x1 - (x2-x1)* 0.05),0)
                    coordinates_arr[e_num][1] = max(int(y1 + (y1-y2) * 0.05), 0)
                    coordinates_arr[e_num][2] = int(x2 + (x2-x1)* 0.05)
                    coordinates_arr[e_num][3] = int(y2 - (y1-y2) * 0.05)
            else:
                return None

            return coordinates_arr


# In[8]:


class Dlib_frontal_face_detector:
    
    def __init__(self):
        # Определяем детектор dlib
        self.detector = dlib.get_frontal_face_detector()
    
    def find_faces(self, img):
        # Конвертируем изображение в оттенки серого
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        # Детектируем лица на изображении
        faces = self.detector(gray)
        # Проверяем условие наличия лица на изображении
        if faces:
            coordinates_arr = np.ones((len(faces),4), dtype='uint16')
            for e_num, face in enumerate(faces):
                # Получаем значения координат расположения лица на изображении
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                
                # Получаем значение высоты и ширины зоны детектирования лица
                hight = y2 - y1
                wight = x2 - x1
                # Расчитываем макмиальную длину стороны прямоугольника
                max_side = max(hight, wight)
                # расчитываем значение для смещения координат по y
                b_y = (hight - max_side) // 2
                # расчитываем значение для смещения координат по x
                b_x = (wight - max_side) // 2
                # Вносим коррективы в значения по оси y
                y1 += b_y
                y2 += b_y
                # Вносим коррективы в значения по оси x
                x1 += b_x
                x2 += b_x
                
                # Сохраняю координаты детектированного лица с корректировкой расширения зоны лица
                coordinates_arr[e_num][0] = max(int(x1 - (x2-x1)* 0.2),0)
                coordinates_arr[e_num][1] = max(int(y1 + (y1-y2) * 0.2), 0)
                coordinates_arr[e_num][2] = int(x2 + (x2-x1)* 0.2)
                coordinates_arr[e_num][3] = int(y2 - (y1-y2) * 0.2)
        else:
            return None
        
        return coordinates_arr


# In[10]:


class VGG_model_classification:
    """
    VGG_model_classification - обученая модель классификации эмоций, на архитектуре ResNet50 или VGG16.
                            ! СТОИТ ПОМНИТЬ О ФУНКЦИИ ПРЕПРОЦЕССИНГА ИЗОБРАЖЕНИЯ!
    
    path_to_model - > путь к обученой модели. 
    
    """
    
    def __init__(self, path_to_model=f'{os.getcwd()}/saved_models/VGG_ResNet50'):
        # Загружаем модель
        # По умолчанию модель ResNet50
        self.base_model = keras.saving.load_model(path_to_model)
        # По умолчанию для ResNEt50
        self.model = self._build_model()
        
        
        # Словарь для вывода предсказаний
        emotion_dict = {'anger': 0,
                        'contempt': 1,
                        'disgust': 2,
                        'fear': 3,
                        'happy': 4,
                        'neutral': 5,
                        'sad': 6,
                        'surprise': 7,
                        'uncertain': 8}
        
        self.emotion_dict = dict((v,k) for k, v in zip(emotion_dict.keys(), emotion_dict.values()))
        
    def _build_model(self):
        
        model = Sequential()
        model.add(self.base_model)
        # Так как мы обучали модели без функции активации, то добавляем функцию активации softmax
        model.add(Activation(activation='softmax'))
        
        return model
    
    def get_emotion(self, img):
        return [self.emotion_dict[i] for i in np.argmax(self.model.predict(img, verbose=0), axis=1)]
    


# In[11]:


class VGG_model_regression:
    """
    VGG_model_regression - обученая модель регрессии эмоций, на архитектуре ResNet50.
    
    path_to_model - > путь к обученой модели. 
    
    """
    
    def __init__(self, path_to_model=f'{os.getcwd()}/saved_models/VGG16_V_A'):
        # Загружаем модель
        # По умолчанию модель ResNet50
        self.base_model = keras.saving.load_model(path_to_model)
        self.model = self._build_model()

        # Словарь для вывода предсказаний
        emotion_dict = {'anger': 0,
                        'contempt': 1,
                        'disgust': 2,
                        'fear': 3,
                        'happy': 4,
                        'neutral': 5,
                        'sad': 6,
                        'surprise': 7,
                        'uncertain': 8}
        
        self.emotion_dict = dict((v,k) for k, v in zip(emotion_dict.keys(), emotion_dict.values()))
        
        # Словарь для вычисления попарных дистанций
        VA_emotions = {
            'anger':    [ 1.5, 6.72],   # гнев
            'contempt': [ 1.5, 3.75],   # презрение
            'disgust':  [ 3.5,  2.5],   # отвращение
            'fear':     [ 2.0,  6.2],   # страх
            'happy':    [ 6.6,  4.2],   # радость
            'neutral':  [ 4.0,  4.0],   # безразличие
            'sad':      [ 1.5,  1.5],   # грусть
            'surprise': [ 6.0,  6.0],   # удивление
            'uncertain':[ 3.0,  4.0]    # неопределенность
        }
        
        self.emotion_arr = np.array(
            [i for k,i in VA_emotions.items()]
        )
        
    def _build_model(self):
        
        model = Sequential()
        model.add(self.base_model)
        # Так как мы обучали модели без функции активации, то добавляем функцию активации linear
        model.add(Activation(activation='linear'))
        return model
    
    
    def get_emotion(self, img):
        # предсказынные 2 значения Valence Arousal
        V_A = self.model.predict(img, verbose=0)
        # Вычисляем близжайший центр кластера
        max_id = np.argmin(pairwise_distances(V_A, self.emotion_arr), axis=1)
        # Возврат текстовых значений
        return [self.emotion_dict[i] for i in max_id]
    


# In[12]:


class Draw_box:
    
    """
    Обьект рисовщик, на вход изображение -> отрисовка необходимых елементов -> итоговое изображение
    
    """
    def __init__(self):
        self.cf_thickness = 87320.25
        
    # Функия рисования box
    def plot_box(self, img=None, coordinates=None, 
                  model_prediction=None, 
                  v_a = None, rectangle_color=None):
        
        
        # Рисуем все детекции на изображении
        # В случае отсутствия координат и предсказания модели возвращаем исходное изображение
        try:
            # Получаем thickness_rate, взависимости от размера изображения
            shape_img = img.shape
            square = shape_img[0] * shape_img[1]
            thickness_rate = min(4, int(square / self.cf_thickness))
            th_2 = min(thickness_rate/2, 1.5)

            for e_num, (i_pack, i_pred)  in enumerate(zip(coordinates, model_prediction)):
                x1, y1, x2, y2 = i_pack
                cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=rectangle_color, thickness=thickness_rate)
                cv2.putText(img, i_pred, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, th_2, (13,13,173), thickness=thickness_rate-1)
                # В случае передачи в функцию значений v_a (Valence Arousal) будем их рисовать
                if v_a is not None:
                    v_a_text = f'Val:{v_a[e_num][0]} Aro:{v_a[e_num][1]}'
                    cv2.putText(img, v_a_text, (x2, y1), cv2.FONT_HERSHEY_DUPLEX, th_2, (13,13,173), thickness=thickness_rate)
            return img
        
        except Exception as ex:
            print(ex)
            return img


# In[ ]:


class Inference:
    
    """
    Обьект Inference - предназначен для стыковки основных компонентов проекта, и определения логики инфиренса:
    
    model_obj       - > обьект модели предсказания эмоций. (По умолчанию наша модель VGG_model);
    face_det_obj    - > обьект детектирования, и возврата координат лиц. (По умолчанию детектор MediaPiPe_detector);
    draw_box_obj    - > обьект отрисовки на изображении результатов предсказания модели. (По умолчанию обьект Draw_box);
    IMG_SIZE        - > размер изображения для входа в модель;
    show_state_mode - > итоговое отображеие результата(применяется лишь для infirence_mode = "img"). По умолчанию cv2. Если режим inline 
                        (для отображения результата в ноутбуке);
    inference_mode  - > принимает по умолчанию режим "img". Предполагает пять режимов:  "img" - для предсказания и отрисовки на изображениях;
                                                                                        "img_batch" -для пакетного предсказания;
                                                                                        "video" - для предсказания и отрисовки на файлах видео;
                                                                                        "video_rec" - для предсказания и отрисовки на файлах видео c последующим сохранением;
                                                                                        "online" - для предсказагния и отрисовки в онлайн режиме-используя веб камеру.
    """
    
    def __init__(self, model_obj=VGG_model_classification(), 
                face_det_obj=MediaPiPe_detector(),
                draw_box_obj=Draw_box(),
                inference_mode = 'img',
                IMG_SIZE=224,
                show_state_mode='cv2'):
        
        self.model_obj = model_obj
        self.face_det_obj = face_det_obj
        self.draw_box_obj = draw_box_obj
        self.inference_mode = inference_mode
        self.img_size = IMG_SIZE
        self.show_state_mode = show_state_mode
        
        
    # процесс получения детекций на одном кадре(координат лиц, и предсказания эмоций)
    def _frame_processing(self, img):
        # Получаем координаты детектированных лиц
        try:
            # Получаю координаты лиц
            coordinates = self.face_det_obj.find_faces(img)
            # батч выходных изображений
            out_images_batch = None
            # Итерируюсь по пакетам координат детектированных лиц
            for i_face_pac in coordinates:
                x1, y1, x2, y2 = i_face_pac
                # Обрезаю лица
                gimg =  np.copy(img[x1:x2, y1:y2])
                 # Преобразую изображение к заданному размеру
                gimg = cv2.resize(gimg, (self.img_size, self.img_size))
                # В случае первой итерации out_images пустой -> создаем исходный
                # батч с первого детектированного лица
                # в противном случае склеиваем все остальные детекции в единый батч
                if out_images_batch is None:
                    out_images_batch = gimg[None, ...]
                else:
                    out_images_batch = np.concatenate((out_images_batch, gimg[None, ::]))
            # Подаем батч изображений в модель предсказания
            prediction_text = self.model_obj.get_emotion(out_images_batch)

            # Возвращаемые значения - координаты детекций лиц на исходном изображении и предсказанные эмоции
            return (coordinates, prediction_text)

        # В случае отсутствия детекций лиц - возвращаем None
        except Exception as ex:
            return (None, None)


    # Инференс для img    
    def _img_inf(self, path_img):
        img = cv2.imread(path_img)
        coordinates, prediction_text = self._frame_processing(img)
        img = self.draw_box_obj.plot_box(img=img,
                                    coordinates=coordinates,
                                    model_prediction=prediction_text,
                                    v_a = None,
                                    rectangle_color=(0, 255, 0))

        if self.show_state_mode=='inline':
            plt.imshow(img[...,::-1],cmap='CMRmap')
            plt.show()
        elif self.show_state_mode=='cv2':
            # Показываем изображение
            cv2.imshow(winname="Face", mat=img)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
            
    
    # Инференс для img    
    def _img_batch_inf(self, path_img):
        id_img = []
        predictions = []
        for i_file in os.listdir(path_img):
            
            if i_file.endswith('.jpg'):
                img = cv2.imread(f'{path_img}/{i_file}')
                try:
                    _, prediction_text = self._frame_processing(img)
                    if prediction_text is None:
                        raise AttributeError
                    else:
                        id_img.append(i_file)
                        predictions.append(prediction_text[0])

                except Exception as ex:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    prediction_text = self.model_obj.get_emotion(img[None, ...])
                    id_img.append(i_file)
                    predictions.append(prediction_text[0])
                    
            else:
                continue
        # Возвращаем 2 списка: id_jpg, предсказания
        return (id_img, predictions)
                                

    # Инференс для online/ либо просмотр видео без сохранения
    def _online_inf(self, path_to_vid=0):
        cap = cv2.VideoCapture(path_to_vid) 
        while True:
            _, frame = cap.read()
            coordinates, prediction_text = self._frame_processing(frame)
            img = self.draw_box_obj.plot_box(img=frame,
                                        coordinates=coordinates,
                                        model_prediction=prediction_text,
                                        v_a = None,
                                        rectangle_color=(0, 255, 0))
            
            # Показываем изображение
            cv2.imshow(winname="Face", mat=img)
            # Exit when escape is pressed
            if cv2.waitKey(delay=1) == 27:
                break
        # When everything done, release the video capture and video write objects
        cap.release()
        # Close all windows
        cv2.destroyAllWindows()


    # Инфиренс с сохранением видео
    def _video_inf(self, path_to_vid=None, path_save_vid='/'):
        # Отктываем путь к видео
        cap = cv2.VideoCapture(path_to_vid)
        cap.set(3,640)
        cap.set(4,480)
        # Загружаем кодеки для записи видео
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(f'{path_save_vid}output.mp4', fourcc, 20.0, (640,480))

        while(True):
            _, frame = cap.read()
            coordinates, prediction_text = self._frame_processing(frame)
            img = self.draw_box_obj.plot_box(img=frame,
                                        coordinates=coordinates,
                                        model_prediction=prediction_text,
                                        v_a = None,
                                        rectangle_color=(0, 255, 0))
            # Сохраняем кадр
            out.write(img)
            # Отображаем кадр
            cv2.imshow('frame', img)
            # В случае нажатия Esc приостанавливаем запись
            if cv2.waitKey(delay=1) == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        
    # Функция - которая измеряет скорость работы модели включая детекцию лиц
    def _benchmark_frame(self, path_to_data=None, iterations=100):
        img = cv2.imread(path_to_data)
        img = cv2.resize(img, (self.img_size, self.img_size))
        #Тестируем скорость с детектированием лиц на изображении
        self._frame_processing(img) # Прогреваем модель
        inference_times = []
        for i in range(iterations):
            start = time.time()
            self._frame_processing(img)
            inference_times.append(time.time() - start)
        mean_time = np.mean(inference_times)
        median_time = np.median(inference_times)
        plt.plot(inference_times)
        plt.title(f'Inference time over {iterations} iterations on one picture\n'
                  f'mean time = {mean_time:.3f}sec, median time = {median_time:.3f}sec')
        plt.xlabel('Iteration number')
        plt.ylabel('Inference time, sec')
        plt.show()
        return mean_time, median_time

    
    # Функция, которая запускает инфиренс модели, взависимости от параметров инициализации
    def go(self, path_to_data=None, path_save_data='/'):
        if self.inference_mode == 'img':
            self._img_inf(path_to_data)
        elif self.inference_mode == 'online':
            self._online_inf()
        elif self.inference_mode == 'video':
            self._online_inf(path_to_data)
        elif self.inference_mode == 'video_rec':
            self._video_inf(path_to_data, path_save_data)
        else:
            print("inference_mode not correctly!")
            


# In[15]:


# Функция которая генерирует модель по определенныым гипер-параметрам
def create_model(base_model=None, 
                 start_units=None, 
                 activation_ch=None, 
                 num_layers_start=None, 
                 dropout_rate=0.25, 
                output_num = 9):
    
    # Определяем полносвязную модель
    model = Sequential()
    # Добавляем базовую модель
    model.add(base_model)
    # Выпрямляем слой
    model.add(Flatten())
    
    # Добавляем полносвязный слой
    model.add(Dense(
        # Tune number of units.
        units=start_units))
    model.add(BatchNormalization()) # Нормализация
    model.add(Activation(activation=activation_ch))
    model.add(Dropout(rate=dropout_rate))

    if num_layers_start > 0:
        # Циклом добавляем количество полносвязных слоев
        for i in range(num_layers_start):
             # Добавляем слой нормализации
            
            model.add(Dense(
                # Tune number of units.
                units=(start_units // (2 * (i+1))))) # Линейное уменьшение количества нейронов(с каждой глубиной уменьшаем в 2 раза)
            model.add(BatchNormalization()) # Нормализация
            model.add(Activation(activation=activation_ch))
            # Добавляем дропаут
            model.add(Dropout(rate=dropout_rate))
        
    # На выходе из модели полносвязный слой на 9 категорий 
    model.add(Dense(output_num)) # Без активации, ускоряем поиск

    return model
    

# Создадим гипер модель для оптимизации гиперпараметров сети
class MyHyperModel(HyperModel):
    
    """
    
    MyHyperModel -  гипермодель, для поиска лучших гиперпараметров сети
    """
    def __init__(self, base_model):
        self.base_model = base_model

    def build(self, hp):
         # Определяем стартовое количество нейронов в полносвязной сети
        start_uits = hp.Int("units_start", min_value=256, max_value=608, step=32) 
        # Определяем функцию активации
        activation_ch = hp.Choice("activation_start", ["relu", "tanh"])
        # Колличество слоев сети
        num_layes_start = hp.Int('num_layers', 1, 3)
        
        model = create_model(
            base_model=self.base_model,
            start_units=start_uits,
            activation_ch=activation_ch, 
            num_layers_start=num_layes_start,
            dropout_rate=0.25
        )
        
        # Компилируем нашу модель
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=keras.optimizers.Adam(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        
        return model

