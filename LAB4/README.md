# **Лаборатораня работа №4 по дисциплине "Системы компьютерного зрения"**

## Задание
В рамках данной работы было необходимо обучить модель с любой архитектурой на наш выбор. Задача модели - находить, определять несколько классов объектов на изображении.

## Выбор датасета 
Был выбран датасет состоящий из фоток самоллетов, кораблей и машин. Датасет содержит по 1000 изображений каждого класса. Датасет не имеет локализационных меток (то есть не имеет координат объектов на изображении, только метки классов).

Ссылка на датасет: [Multiclass-image-dataset-airplane-car-ship](https://www.kaggle.com/datasets/abtabm/multiclassimagedatasetairplanecar/data)

## Выбор архитектуры модели
Выбор модели для задачи классификации состоит из двух направлений:

1) Использование уже готовой архитектуры

2) Создание своей архитектуры

Рассмотрим первый случай. Он заключается в выборе каких-либо готовых моделей. В частности, будет необходим выбор архитектуры непосредственно для задачи классификации транспорта, среди самых популярных можно отметить:

1) ResNet-18/50
    - Подходит для быстрой и стабильной классификации, включая transfer learning
    - Хорошо работает на большом и среднем датасете

2) EfficientNet B0-B3 (Одновременное масштабирование ширины, глубины и разрешения (compound scaling))
    - Высокая точность при малом числе параметров и FLOPs
    - Compound scaling позволяет сбалансированно масштабировать архитектуру
    - Требует точной настройки входного разрешения
      
3) VGGNet (Глубокая, но однородная архитектура: все свертки 3×3, пулы 2×2.)
    - Простая, последовательная архитектура
    - Имеет много параметров (≈138 млн), высокая вычислительная сложность
      
4) DenseNet (Полная связанность — каждый слой получает входы от всех предыдущих, эффективное переиспользование признаков)
    - Регуляризация через сохранение идентичности. Эффективная обрабатка исходных данных на каждом слое без переобучения.
    - Устойчивость к переобучению из-за глубокого потока градиентов
    - Большое потребление памяти: большое количество feature maps передаются между слоями
  
Все перечисленные архитектуры предобучены на датасете **ImageNet-1k**, включающий такие классы:
- Car: race car, racing car, freight car, passenger car;
- Ship: container ship, pirate ship, aircraft carrier, attack aircraft carrier, fireboat, liner, ocean liner;
- Airplanes: airliner, plane, warplane, military plane.

На основе выше указанной информации была выбрана архитектура ResNet18, так как она является одной из наиболе быстродейсвующих с достаточно высокой точностью.

### Описание архитектуры ResNet
ResNet (сокр. от Residual Network) — это архитектура глубоких нейронных сетей, предложенная в 2015 году исследователями из Microsoft Research. Она стала прорывом в области обучения сверхглубоких сетей (сотни слоёв) благодаря введению остаточных соединений (residual connections).

С момента публикации ResNet-50 стал де-факто стандартом для классификационных задач, особенно в задачах transfer learning. Семейство моделей включает ResNet-18, 34, 50, 101, 152 и более глубокие варианты.

## Архитектура модели ResNet

Backbone: ResNet18  

Основная сеть для извлечения признаков, построенная на остаточных блоках. 
1) Входной слой    
     - **Conv7x7**: Свёрточный слой с фильтром 7×7, 64 каналами, шагом 2.  
     - **MaxPool3x3**: Слой подвыборки с окном 3×3 и шагом 2.  
     - **BatchNorm2d**: Нормализация данных для ускорения обучения и стабилизации.  
     - **ReLU**: Функция активации.
     
3) Этапы (Stages) Модель разделена на 4 этапа, каждый из которых содержит 2 остаточных блока типа BasicBlock (для ResNet18/34).   
    
  	- Stage 1:   
    	- Размер карты признаков: 56×56.  
    	- 2 блока с 64 каналами.
    	- Прямые соединения (skip-connections) внутри блоков.
         
	- Stage 2 :   
  		- Downsample: Уменьшение размера карты признаков до 28×28 через свёртку 1×1 с шагом 2.  
  		- 2 блока с 128 каналами.     

	- Stage 3 :   
	    - Downsample до 14×14.  
	    - 2 блока с 256 каналами.     

	- Stage 4 :   
	    - Downsample до 7×7.  
	    - 2 блока с 512 каналами.

4) Финальные слои
   
     - **AdaptiveAvgPool2d**: Среднее усреднение до фиксированного размера 1×1.  
     - **Fully Connected (FC)**: Полносвязный слой с 1000 выходами (для ImageNet) или другим количеством классов.
     
**Структура представлена ниже**

![image001](https://github.com/user-attachments/assets/3fff6866-0bb2-4f58-8efe-f45defd1a861)


## Особенности обучения модели

### Функции потерь  

- CrossEntropyLoss : Комбинация Softmax и NLLLoss для многоклассовой классификации.  
- Label Smoothing : Сглаживание меток для улучшения обобщаемости.
     
### Аугментации данных  

- RandomResizedCrop : Случайное обрезание изображения до 224×224.
- RandomHorizontalFlip : Горизонтальное отражение.
- RandomRotation : Случайный поворот.
- ColorJitter : Изменение яркости, контраста, насыщенности.
- Normalization : Нормализация по ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

## Подготовка к обучению

Датасет подгружается с Kagglehub. Датасет изначально разбит на две выборки обучающая(train) и тестовую(test). Для валидации в процессе обучения обучающая выборка разделена на 2 части с отношениями: 80% - обучающая (train), 20% - валидационная (valid). 

Чтобы улучшить обобщающую способность модели, сделать её устойчивой к различиям в реальных данных и избежать переобучения на тренировочных примерах произведем аугментацию и нормализацию данных.

## Обучение модели

Основными параметрами модели являются:

- Размер батча (batch_size) определяет, сколько примеров данных используется для обновления весов модели за один шаг градиентного спуска.
- Число эпох (epochs) определяет, сколько раз модель будет проходить через весь набор данных во время обучения, главное, чтобы модель не переобучилась.
- Скорость обучения (Learning Rate) определяет, насколько сильно обновляются веса модели  после каждой итерации. Он управляет "шагом", с которым модель движется по поверхности функции потерь в поисках минимума. Используется оптимизатор Adam c планировщиком скорости обучения ReduceLROnPlateau.

### 1) Первый вариант обучения на предобученной моделе на датасете ImageNet.

Кривые обучения

![Без имени](https://github.com/user-attachments/assets/627c8c1c-8bf7-4306-9c83-0ffa70e9bc2b)

Оценочные метрики на тестовой выборке
<pre>
Classification Report:
              precision    recall  f1-score   support

   airplanes       1.00      0.99      0.99       189
        cars       0.99      1.00      0.99       193
        ship       1.00      1.00      1.00       200

    accuracy                           1.00       582
   macro avg       1.00      1.00      1.00       582
weighted avg       1.00      1.00      1.00       582</pre>

Матрица ошибок (Confusion Matrix)

![Без имени](https://github.com/user-attachments/assets/a7958a00-ae52-4422-bd74-40ff58790a05)

Графики показывают менее стабильное поведение: колебания на валидации на высоких значениях точности. Некоторые скачки val_loss могут намекать на переобучение. 
Предобученная модель ResNet демонстрирует идеальные результаты на тестовых данных. Высокие значения precision, recall, F1-score и accuracy указывают на то, что модель практически безошибочно классифицирует все объекты.

### 2) Второй вариант обучения без предобучения.

Кривые обучения

![Без имени](https://github.com/user-attachments/assets/754e339c-81e8-402d-8b99-f7c2559c57a5)

Оценочные метрики на тестовой выборке
<pre>
Classification Report:
              precision    recall  f1-score   support

   airplanes       0.93      0.92      0.92       189
        cars       0.93      0.95      0.94       193
        ship       0.97      0.96      0.97       200

    accuracy                           0.94       582
   macro avg       0.94      0.94      0.94       582
weighted avg       0.94      0.94      0.94       582</pre>

Матрица ошибок (Confusion Matrix)

![Без имени](https://github.com/user-attachments/assets/825e0f10-ebf0-4913-a007-45780858fc63)

Модель ResNet демонстрирует высокую точность на тестовой выборке, с общей точностью классификации 94%.
Модель хорошо различает все три класса, особенно хорошо справившись с классификацией кораблей (precision = 0.97, recall = 0.96).
Графики потерь и точности подтверждают, что модель успешно обучается и обладает хорошей обобщающей способностью, так как валидационная точность достигает 95%  без значительного разрыва между тренировочной и валидационной потерями.

Для задачи классификации изображений самолетов, машин и кораблей предобученная модель ResNet является наиболее эффективным решением, обеспечивающим почти идеальную точность даже на ограниченном датасете.

Обученные модели сохранены в этом репозитории (model_pred.pth, model_nopred.pth)

## Тесты на своих изображениях

Был выбран ряд случайных изображений для проверки работы незнакомых для модели.

<img src="https://github.com/user-attachments/assets/c8e83c08-f967-4649-b176-6d18399be924" width="400" />
<img src="https://github.com/user-attachments/assets/941a94ac-5f63-4a5d-9eb7-9c04d269a53b" width="400" />
<img src="https://github.com/user-attachments/assets/40568ee6-0fd1-479d-9dff-95a96aba4032" width="400" />
<img src="https://github.com/user-attachments/assets/4dcdbad7-b3ee-4c0c-a39c-6d5548b26dbd" width="400" />
<img src="https://github.com/user-attachments/assets/ae911721-553d-4854-97f7-cafd871998a5" width="400" />
<img src="https://github.com/user-attachments/assets/54f3364e-1e37-40c6-8818-751e3880ab63" width="400" />

## Итоги лабораторной работы

  В рамках данной лабораторной работы были выбран датсет и использован для обучения модели на основе архитектуры ResNet, после была проведена оценка работы модели на тестовых данных, получены основные метрики модели и приведена их оценка, которая показал высокую точность модели на простых изображениях.
