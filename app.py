from flask import Flask, render_template, request, redirect
from whiletruepredict import FolderProcessor, THRESH
import joblib
import json


app = Flask(__name__)

# заглушка
dates = "null"


@app.route('/', methods=['POST', 'GET'])
@app.route('/card')
def home():
    if request.method == 'POST':
        if request.form.get('form') == 'path':
            path = request.form.get('path')
            predictor = joblib.load('models/lg400.jbl')
            output = 'data/data.json' #куда пишуться данные
            fp = FolderProcessor(predictor, THRESH)
            datas = fp.process_dir(path)  #путь к папке
            with open(output, 'w') as fout:  
                json.dump(datas, fout)
            return redirect('/')
        data_1 = request.form.get('date_1')
        data_2 = request.form.get('date_2')
        return str(data_1) + " " + str(data_2)
    return render_template('home.html', date=data())


@app.route('/card/<timestamp>')
def date(timestamp):
    return render_template('cards.html', timestamp = timestamp, data=data())


# вкладка для полного отображения
@app.route('/card/<name>/map')
def map(name):
    return render_template('map_date.html', name=name, data=data())


def mes(message):
    return render_template('error.html', message=message)

def data():
    with open('data/data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


if __name__ == '__main__':
    app.run(debug=True, port='2451', host='0.0.0.0')