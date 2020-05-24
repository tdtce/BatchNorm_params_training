# BatchNorm_params_training
Изучение влияния параметров BatchNorm на точность. Реализация статьи [Training BatchNorm and Only BatchNorm:
On the Expressive Power of Random Features in CNNs](https://arxiv.org/pdf/2003.00152.pdf)

## Установка
Пакеты необходимые для работы
```
Python          3.7.1
numpy           1.18.1
tensorboard     2.1.1
torch           1.4.0
torchvision     0.5.0
tqdm            4.43.0
```
Если необходима поддержка GPU, то установка с [сайта](https://pytorch.org/), для установки без GPU используй:
```shell
git clone https://github.com/tdtce/BatchNorm_params_training
cd ./BatchNorm_params_training
pip install -r requirements.txt
```

Веса для моделей можно скачать [здесь](https://drive.google.com/file/d/1CqJVSkHknjAAp7hvVDsaVZTbfNU-bmUY/view).

## Отчет
Отчет по работе [здесь](https://github.com/tdtce/BatchNorm_params_training/blob/master/summary.ipynb).
Несколько примеров по работе фреймворка [здесь](https://github.com/tdtce/BatchNorm_params_training/blob/master/examples.ipynb).

## Общая структура фреймворка
```
repository
    └── runner.py                              # Основной скрипт, строит модель, запускает тренировку или тестирование
    └── dataloader
    │       └── dataloader.py                  
    │       └── transforms.py                  
    └── engine
    │       └── test.py                        # Содержит код для обучения и валидации
    │       └── train.py                       # Здесь для тестирования
    └── models
    │       └── builder.py                     # Создание модели по названию
    │       └── resnet_cifar.py                # Модель как в статье - специально приготовленный ResNet
    │       └── utils.py                       # Утилиты для создания моделей
    └── utils
    │       └── logger.py                      # Логгирование ошибки и точности, вывод в тензорборд и консоль
    │       └── utils.py                       # Утилиты для разных модулей
    │       └── visual.py                      # Функции для отрисовки графиков
    │                                          # Следующие папки и файлы добавлены для показа результатов 
    └── Weight                                 
    │       └── runs                           # Содержит информацию о тренировке
    └── predictions        
            └── save_results                   # Результаты моделей на тесте и предсказания 
    └── img                                    # Изображения для отчета       
```

## Использование
Взаимодействие происходит через файл runner.py. Он принимает параметры из командной строки.
```
Возможные параметры:
    --name           : название эксперимента, используется для сохранения.
    --model          : название архитектуры сети, оно дальше парсится и создается 
                       сеть. Если указан неверно выбросить исключение.
    --train          : включается режим тренировки. Если не указан, то по умолчанию 
                       режим тестирования.
    --epoch          : количество эпох для тренировки.
    --batch_size     : размер батча.
    --learning_rate  : начальный learning rate.
    --weight_path    : путь для весов, используются для тестирования. По умолчанию 
                       используется вес с префиксом baseline. Если указать init, 
                       используются веса объявленные при инициализации. 
    --gpu            : ключ позволяющий тренировать на видеокарте. Если не указать, 
                       то CPU.
    --data_dir       : папка для загрузки датасета. По умолчанию ~.
```

Команда для обучения с параметрами как в статье
```
python runner.py --train --epoch 160 --batch-size 128 --learning-rate 0.1 --model cifar_resnet_110
```
Пример команды для тестирования без обученных весов
```
python runner.py  --batch-size 128 --model cifar_freeze_14_2 --weight-path "init"
```

## Доступные модели
Создание модели происходит с помощью параметра --model. 
Для использования доступны все модели из статьи:
- cifar_resnet_14
- cifar_resnet_32
- cifar_resnet_56
- cifar_resnet_110
- cifar_resnet_218
- cifar_resnet_434
- cifar_resnet_866
- cifar_resnet_14_1
- cifar_resnet_14_2
- cifar_resnet_14_4
- cifar_resnet_14_8
- cifar_resnet_14_16
- cifar_resnet_14_32
- cifar_freeze_14
- cifar_freeze_32
- cifar_freeze_56
- cifar_freeze_110
- cifar_freeze_218
- cifar_freeze_434
- cifar_freeze_866
- cifar_freeze_14_1
- cifar_freeze_14_2
- cifar_freeze_14_4
- cifar_freeze_14_8
- cifar_freeze_14_16            
- cifar_freeze_14_32

Первое число отвечает за общее количество слоев, вторая это множитель для количества весов на слое. Если второе число не указано значит множитель 1. С помощью первого числа можно увеличивать сеть в глубину, а с помощью второго - вширину. freeze в названии значит, что обучаются только веса и смещения для BatchNorm.

## Визуализация
Примеры построения графиков можно увидеть в [отчете](https://github.com/tdtce/BatchNorm_params_training/blob/master/summary.ipynb). В процессе обучения точность и ошибка сохраняются в csv формате, а также в tensorboard формате. Для запуска tensorboard:
```
tensorboard --logdir runs/{name}
```
Графики можно увидеть по адресу http://localhost:6006/.
