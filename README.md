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

## Фреймворк для обучения
Общая структура фреймворка
```
repository
    └── runner.py                              # Основной скрипт, строит модель, запускает тренировку или тестирование
    └── dataloader
    │       └── dataloader.py                  
    │       └── transforms.py                  
    └── engine
    │       └── test.py                        # Содержит код для обучения и валидации
    │       └── train.py                       # Здесь для тестировани
    └── models
    │       └── builder.py                     # Создание модели по названию
    │       └── resnet_cifar.py                # Модель как в статье - специально приготовленный ResNet
    └── utils
            └── logger.py                      # Логгирование ошибки и точности, вывод в тензорборд и консоль
            └── utils.py                       # Утилиты для разных модулей
```

Взаимодействие происходит через файл runner.py. Он принимает параметры из командной строки.
```
Возможные параметры:
    --name           : название эксперимента, используется для сохранения.
    --model          : название архитектуры сети, оно дальше парсится и создается 
                       сеть. Если указан неверно выбросить исключение.
    --test           : включается режим тестирования. Если не указан, то по умолчанию 
                       режим тренировки.
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
Примеры команды для тестирования без обученных весов
```
python runner.py  --batch-size 128 --model cifar_freeze_14_2 --weight-path "init"
```
