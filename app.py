from flask import Flask, render_template, request, redirect
from whiletruepredict import FolderProcessor, THRESH
import joblib
import json


app = Flask(__name__)

# заглушка
dates = {"19.03.2005" : [["photo.jpg", "name-1", "19.03.2005", "72, 62", "34"], ["photo.jpg", "name-1", "19.03.2005", "72, 62", "34"], ["photo.jpg", "name-1", "19.03.2005", "72, 62", "34"]],
            "12.10.2020" : [["photo.jpg", "name-1", "19.03.2005", "72, 62", "34"], ["photo.jpg", "name-1", "19.03.2005", "72, 62", "34"], ["photo.jpg", "name-1", "19.03.2005", "72, 62", "34"]]}


@app.route('/', methods=['POST', 'GET'])
@app.route('/card')
def home():
    if request.method == 'POST':
        # дата от какого числа отчёт
        data_1 = request.form.get('date_1')
        data_2 = request.form.get('date_2')
        
        predictor = joblib.load('models/lg400.jbl')
        output = 'result.json' #куда пишуться данные
        fp = FolderProcessor(predictor, THRESH)
        datas = fp.process_dir(path)  #путь к папке
        with open(output, 'w') as fout:  
            json.dump(datas, fout)
        
        
        return str(data_1) + " " + str(data_2)
    return render_template('home.html', date=dates)


@app.route('/card/<name>')
def date(name):
    return render_template('cards.html', res=dates[name])


# вкладка для полного отображения
# @app.route('/card/<name>/map')
# def map(name):
#     return render_template('map_date.html', res=dates)


def mes(message):
    return render_template('error.html', message=message)


if __name__ == '__main__':
    app.run(debug=True, port='2451', host='0.0.0.0')