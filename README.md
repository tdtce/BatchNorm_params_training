# BatchNorm_params_training
Изучение влияния параметров BatchNorm на точность. Реализация статьи [Training BatchNorm and Only BatchNorm:
On the Expressive Power of Random Features in CNNs](https://arxiv.org/pdf/2003.00152.pdf)

## Установка
Пакеты необходимые для работы
```shell
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
