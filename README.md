<p align="center">
    <h1 align="center">Секретный проект</h1>
    </p>
<p>Оставлю тут название своей команды WhileTrue</p>
<p>Может быть вы слышали о ней</p>

<h4>Реализованная функциональность</h4>
<ul>
    <li>Обработка 13-канальных изображений</li>
    <li>Вывод основной информации на сайте в удобном формате</li>
    <li>Подсчет кв. метров, пораженных нефтью</li>
</ul> 
<h4>Особенность проекта в следующем:</h4>
<ul>
 <li>Возможность дообучения на большем кол-ве данных</li>
 <li>Сегментация классических изображений в формате tiff, jpg, png и т.д.</li>
 <li>Обработка изображений в режиме реального времени</li>  
 </ul>
<h4>Основной стек технологий:</h4>
<ul>
	<li>HTML, JavaScript, Flask</li>
	<li>PHP 7, MySQL.</li>
	<li>Git</li>
	<li>Python, Pytorch, EO learn.</li>
  
 </ul>
<h4>Демо</h4>
<p>Тут ссылка на гугл диск </p>
<p>Пока в формате видео</b></p>




СРЕДА ЗАПУСКА
------------
1) необходимо установить miniconda3 или anaconda3;
2) создать новое окружение;
3) установаить все библиотеки из requirements.txt
4) запустить.


УСТАНОВКА
------------
### Установка пакета name

Выполните 
~~~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda create -n newenv
git clone https://github.com/Demontego/whileTrue_new_case
cd whileTrue_new_case
pip install -r requirements.txt
...
~~~

После этого выполнить команду в директории проекта:
для формирования json с данными( в будущем это переедет в PostgreSQL
~~~
python whiletruepredict.py 2019-01/ out.json models/lg400.jbl
~~~
Для проверки нейронности сети для сегментации(на сайт данный функционал не добавлен, в дальнейшем можно прикрутить)
~~~
python Segmentation.py images result models/best_model_LinkNet34.pth
~~~
Для запуска сайта с демонстрацией
~~~
python app.py
~~~
Затем надо будет вставить путь к папке которую хотим проанализировать(например 2019-01), после будет доступна информация

РАЗРАБОТЧИКИ

<h4>Курбанов Ринат DS https://t.me/@demontego</h4>
