# Создание и развертывание среды разработки

## Дистрибутив `Anaconda`

[Ссылка на репозиторий](https://repo.anaconda.com/archive/)

Загрузите и установите дистрибутив `Anaconda` в  соответствии с ОС вашего хоста.

## Образ c Ubuntu под VirtualBox

[Ссылка на образ](https://disk.yandex.ru/d/0Hd92rzNB0_IHg)

Образ в формате `.iso` и предназначен для установки под `VirtualBox`

Установленное ПО:

- Ubuntu 18
- Java (JDK) 8
- **Anaconda3-2019.03** (python 3.7)
- IntellJ 2020.2.1
- Hadoop 3.1.2
- Spark 2.4.6
- Zookeeper 3.5.8
- Kafka 2.5.1
- Redis 6.0.7

Пароль: `ubuntu`

Для доступа к общим каталогам на виртуалке необходимо добавить пользователя ubuntu в группу `vboxsf`:

```bash
sudo adduser $USER vboxsf
```

