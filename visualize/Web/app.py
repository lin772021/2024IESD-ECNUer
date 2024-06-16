from flask import Flask, render_template, jsonify, redirect, url_for
import matplotlib.pyplot as plt
from data_load import *
from datetime import timedelta

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
# parameters
SIZE = 1250
path_data = '../../../data/training_dataset/'
path_indices = '../data_indices/'
path_pred = '../predict_data/'
X_test, y_test, filename_list = loadData(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE) # 这是load了全部数据
CNN_pred = loadPred(root_dir=path_pred, name='CNN')
LSTM_pred = loadPred(root_dir=path_pred, name='LSTM')
cnt = 490 # 表示当前计算的是哪个数据

@app.route('/')
def index():
    global cnt
    features = extract_all_features(X_test[cnt])
    print(features)
    return render_template('index.html', data=X_test[cnt].tolist(), filename=filename_list[cnt], y_true = y_test[cnt], lstm_pred=LSTM_pred[cnt], cnn_pred=CNN_pred[cnt], features=features)

@app.route('/increment_cnt', methods=['POST'])
def increment_index():
    global cnt
    cnt += 1
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
