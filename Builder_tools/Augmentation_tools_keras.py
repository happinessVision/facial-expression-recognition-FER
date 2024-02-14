#!/usr/bin/env python
# coding: utf-8

# In[6]:


### Импорт необходимых библиотек
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RandomBrightness, RandomContrast, RandomFlip, RandomRotation, RandomZoom
from PIL import Image
import cv2


# In[2]:


class Generate_img_model:
    '''
       Класс Generate_img_model - это модель нейронной сети, которая применяет аугментации к входному изображению.
       
       dict_random_factor - > параметры для аугментаций.
       
    '''
    def __init__(self, dict_random_factor=None):
        super().__init__()
        self.dict_random_factor = dict_random_factor
        # В случае отсутствия параметров для преобразования изображения, они инициализируются стандартными значениями 
        if self.dict_random_factor is None:
            self.dict_random_factor = {'RandomBrightness':0.3,
                                       'RandomContrast':0.3,
                                       'RandomRotation': 0.15,
                                       'RandomZoom': (-0.2, 0.2)
                                       }
        #  Инициализирую модель для генерации изображений
        self.model = self._build_model()
    # Строю модель
    def _build_model(self):
        # Для генерации новых тестовых данных буду использовать слои TF, со значениями случайных изменений изображений
        #  Создаю модель для генерации новых изображений для обучающего датасета
        model = tf.keras.Sequential()
        # Случайное применение яркости к изображению
        model.add(RandomBrightness(
           self. dict_random_factor['RandomBrightness']))
        # Случайное применение контраста  к изображению
        model.add(RandomContrast(
            self.dict_random_factor['RandomContrast']))
        # Слой предварительной обработки, который случайным образом переворачивает изображения во время обучения.
        model.add(RandomFlip(
            mode="horizontal_and_vertical"))
        # Слой предварительной обработки, который случайным образом поворачивает изображения во время обучения.
        model.add(RandomRotation(
            self.dict_random_factor['RandomRotation']))

        # Слой предварительной обработки, который случайным образом масштабирует изображения во время обучения.
        model.add(RandomZoom(
            self.dict_random_factor['RandomZoom'])) 
        # generate_model_class.add(tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE)) # Изменение размера изображения
        return model
    
    # Прогон изображения --> получение нового искаженного изображения
    def generate_img(self, img):
        return self.model(img[None, ...]).numpy()[0]


# In[3]:


class SMOTE:
    
    """
    Обьект SMOTE - метод увеличения числа примеров миноритарного класса (в нашем случае изображений)
                   (Synthetic Minority Over-sampling Technique, SMOTE) 
                   — это алгоритм предварительной обработки данных, 
                   используемый для устранения дисбаланса классов в наборе данных. 
                    
                   
                   
    path_folder_train_data - > путь к исходному набору данных;
    path_save_imgs         - > путь к выходному набору новых данных;
    smote_value            - > количество выходных экземпляров по каждому из класов;
    gen_model              - > модель для аугментаций изображений;
    face_detector          - > обьект детектора лиц;
    IMAGE_SIZE             - > размер выходных изображений;
    train_part             - > доля тренировочных данных, которые не будут учавствовать в процедуре аугментации;
    modify_coef            - > коэфициент регулирующий количество генераций новых изображений,
                             по умолчанию 1. В случае необходимости частичной генерации изображений, % от smote_value, можно установить
                             значение от 0 - без аугментаций до 1-го аугментации до предельного значения smote_value
    """
    
    def __init__(self, path_folder_train_data='/train', 
                 path_save_imgs='/new_train',
                 smote_value=5000, 
                 gen_model=None, 
                 face_detector=None, 
                 IMAGE_SIZE = 224, 
                 train_part=0.15,
                 modify_coef = 1):
        
        self.path_folder_train_data = path_folder_train_data
        self.path_save_imgs =path_save_imgs
        self.smote_value = smote_value
        self.gen_model = gen_model
        self.detector = face_detector
        self.img_size = IMAGE_SIZE
        self.multi_faces_list = []
        self.bad_img_list = []
        self.hight_train_index = {}
        self.train_part = train_part
        self.modify_coef = modify_coef
    
    # Функция для понижающего и повышающего отбора
    # step_info - > вывод информмции на экран после N шага
    def smote(self, step_info=100):
        
        """ СТАРТОВЫЕ ДАННЫЕ """
        # Подсчет количества классов и вхождений в него
        classes_start_count = {}
        classes_start_names = {}
        for i_fold in os.listdir(self.path_folder_train_data):
            # Получаю названия файлов
            names = [i_name for i_name in os.listdir(self.path_folder_train_data + i_fold) if i_name.endswith('jpg')]
            classes_start_names[i_fold] = names
            classes_start_count[i_fold] = len(names)
        
        
        # Значения необходимых генераций новых изображений с учетом modify_coef
        classes_smote_count = {k: int(max(self.smote_value - i, 0) * self.modify_coef) for k,i in classes_start_count.items()}
        
        #Корректированые выходные значения классов с учетом заданного smote_value. Если smote_value > стартового значения то остается итоговое значение
        # Если smote_value < стартового значения то, остается smote_value
        classes_correct_s_value = {k:int(i) if (i<=self.smote_value) else int(self.smote_value) for k,i in classes_start_count.items()}
        
        # Итоговое количество значений на выходе
        classes_out = {k: int(classes_smote_count[k] + classes_correct_s_value[k]) for k in classes_start_count.keys()}
        
        # Итоговые значения нижнего индекса, по которому в дальнейшем из files будет отобрана часть тренировочных данных
        # Которая не будет учавствовать в аугментациях
        classes_idx = {k: int(i * self.train_part) for k,i in classes_out.items()}
        
        # Присваиваю значение верхнего индекса для дальнейшего разбиения изображений на тренировочные и тестовые         
        self.hight_train_index = {k: int(i * (1-self.train_part)) for k,i in classes_out.items()}
        
        """ ЗАПУСК ИТЕРАЦИЙ ПО КЛАССАМ """
        
        for i_class in classes_start_count.keys():
            if os.path.exists(f'{self.path_save_imgs}/{i_class}/'):
                pass
            else:
                os.makedirs(f'{self.path_save_imgs}/{i_class}/')
                
            
            # Получаю перечень названий изображений в папке:
            files = classes_start_names[i_class]
            # При этом случайным образом перемешиваю их, для обеспечения фактора случайности при проведении повышающего/понижающего отбора
            random.shuffle(files)
            
            """ПРОХОД БЕЗ АУГМЕНТАЦИИ """
            
            # Получаем значение, которое будем использовать для названий обработанных изображений
            name_img_to_save = classes_out[i_class]
            
            # Считаем количество удачных шагов (удачные детекции, отсутствие мультидетекций)
            good_step = 0
            # Считаем количество шагов включая неудачные
            step = 0
            # Итерируемся по изображениям, учитываем то, что ранее они уже перемешаны
            for i_file in files:
                # В случае равенства количества хороших ходов и корректированого значения
                # итерации прерываются, и происходит переход на генерирование новых изображений
                if good_step >= classes_correct_s_value[i_class]:
                    break
                step +=1
                # открываем изображение 
                img = cv2.imread(f'{self.path_folder_train_data}{i_class}/{i_file}')

                # В случае отсутствия детектированных лиц, продолжаем цикл
                try:
                    # Получаю координаты лиц
                    coordinates = self.detector.find_faces(img)
                    # В случае детектирования нескольких лиц на изображении
                    # в целях повышения качества обучающих данных - мы не будем использовать данное изображение
                    if len(coordinates) > 1:
                        self.multi_faces_list.append((i_class, i_file))
                        continue
                        
                    else:
                        x1, y1, x2, y2 = coordinates[0]
                        # Обрезаю лица
                        gimg =  np.copy(img[x1:x2, y1:y2])
                         # Преобразую изображение к заданному размеру
                        gimg = cv2.resize(gimg, (self.img_size, self.img_size))
                        # Сохраняю детектированное лицо
                        cv2.imwrite(f'{self.path_save_imgs}/{i_class}/{name_img_to_save}.jpg', gimg)
                        # Уменьшаю счетчик имен
                        name_img_to_save -= 1
                        good_step += 1
                        if name_img_to_save % step_info == 0:
                            print(f'Save img # {name_img_to_save}, class {i_class}')

                except Exception as ex:
                    self.bad_img_list.append((i_class, i_file))
                    continue
            
            
            """ ПРОХОД С АУГМЕНТАЦИЕЙ  """
            idx_test = int(classes_idx[i_class] + (step - good_step))
            if classes_idx[i_class] / len(files) > 0.5:
                print("TEST_EXAMPLES MORE THEN 0.5 DATA!!!!")
            
            try:
                # Исключаю из перечня изображений те, которые будут в дальнейшем определены как тестовые
                files = files[idx_test+1:]
            except Exception as exc:
                print(exc)
                print(f'LEN_FILES :{len(files)}')
                print(f'LEN_IDX :{idx_test}')
                # В редких случаях может быть такое, что данные будут очень загрязнены
                # тогда, просто возвращаем первоначальный idx
                files = files[classes_idx[i_class]+1:] 
                
            # Итерируюсь до тех пор пока значение остатка не достигнет 0
            while name_img_to_save > 0:
                # Случайным образом выбираю название файла в files и подаю его на вход модели преобразования изображений
                random_img = random.choice(files)
                # Открываю изображение
                img = cv2.imread(f'{self.path_folder_train_data}{i_class}/{random_img}')
                try:
                    # Конвертирую изображение в формат RGB
                    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
                    # Запускаю модель генерации изображений
                    img = self.gen_model.generate_img(img)
                    # Конвертирую обратно в BGR
                    img = cv2.cvtColor(src=img, code=cv2.COLOR_RGB2BGR)
                    # Сохраняю временное изображение
                    cv2.imwrite(f'{self.path_folder_train_data}/temp_img.jpg', img)
                    # Загружаю временное изображение
                    img = cv2.imread(f'{self.path_folder_train_data}/temp_img.jpg')
                except Exception as ex:
                    continue
                # В случае отсутствия детектированных лиц, продолжаем цикл
                try:
                    # Получаю координаты лиц
                    coordinates = self.detector.find_faces(img)
                     # В случае детектирования нескольких лиц на изображении
                    # в целях повышения качества обучающих данных - мы не будем использовать данное изображение
                    if len(coordinates) > 1:
                        continue
                        
                    else:
                        x1, y1, x2, y2 = coordinates[0]
                        # Обрезаю лица
                        gimg =  np.copy(img[x1:x2, y1:y2])
                        # Преобразую изображение к заданному размеру
                        gimg = cv2.resize(gimg, (self.img_size, self.img_size))
                        # Сохраняю детектированное лицо
                        cv2.imwrite(f'{self.path_save_imgs}/{i_class}/{name_img_to_save}.jpg', gimg)
                        # Уменьшаю счетчик имен
                        name_img_to_save -= 1
                        if name_img_to_save % step_info == 0:
                            print(f'Save img # {name_img_to_save}, class {i_class}')

                except Exception as ex:
                    continue
                finally:
                    os.remove(f'{self.path_folder_train_data}/temp_img.jpg')


# In[ ]:




