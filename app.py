from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, jsonify, render_template
from subprocess import call
from flask_socketio import SocketIO, send


app = Flask(__name__)
app.secret_key = "mysecret"
socket_io = SocketIO(app)

# 예측 모델 로드
from tensorflow.python.keras.models import load_model
import numpy as np

rows = np.loadtxt("./lotto.csv", delimiter=",")
row_count = len(rows)
# print(row_count)

import numpy as np

# 당첨번호를 원핫인코딩벡터(ohbin)으로 변환
def numbers2ohbin(numbers):

    ohbin = np.zeros(45) #45개의 빈 칸을 만듬

    for i in range(6): #여섯개의 당첨번호에 대해서 반복함
        ohbin[int(numbers[i])-1] = 1 #로또번호가 1부터 시작하지만 벡터의 인덱스 시작은 0부터 시작하므로 1을 뺌
    
    return ohbin

# 원핫인코딩벡터(ohbin)를 번호로 변환
def ohbin2numbers(ohbin):

    numbers = []
    
    for i in range(len(ohbin)):
        if ohbin[i] == 1.0: # 1.0으로 설정되어 있으면 해당 번호를 반환값에 추가한다.
            numbers.append(i+1)
    
    return numbers


numbers = rows[:, 1:7]
ohbins = list(map(numbers2ohbin, numbers))

x_samples = ohbins[0:row_count-1]
y_samples = ohbins[1:row_count]


train_idx = (0, 800)
val_idx = (801, 900)
test_idx = (901, len(x_samples))

# print("train: {0}, val: {1}, test: {2}".format(train_idx, val_idx, test_idx))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# 88회부터 지금까지 1등부터 5등까지 상금의 평균낸다.
mean_prize = [ np.mean(rows[87:, 8]),
           np.mean(rows[87:, 9]),
           np.mean(rows[87:, 10]),
           np.mean(rows[87:, 11]),
           np.mean(rows[87:, 12])]

print(mean_prize)

def gen_numbers_from_probability(nums_prob):

    ball_box = []

    for n in range(45):
        ball_count = int(nums_prob[n] * 100 + 1)
        ball = np.full((ball_count), n+1) #1부터 시작
        ball_box += list(ball)

    selected_balls = []

    while True:
        
        if len(selected_balls) == 6:
            break
        
        ball_index = np.random.randint(len(ball_box), size=1)[0]
        ball = ball_box[ball_index]

        if ball not in selected_balls:
            selected_balls.append(ball)

    return selected_balls
train_total_reward = []
train_total_grade = np.zeros(6, dtype=int)

val_total_reward = []
val_total_grade = np.zeros(6, dtype=int)

test_total_reward = []
test_total_grade = np.zeros(6, dtype=int)

model = keras.models.load_model('./model_0100.h5') 

model.reset_states()

# print('receive numbers')

xs = x_samples[-1].reshape(1, 1, 45)
ys_pred = model.predict_on_batch(xs)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat')
def chatting():
    return render_template('chat.html')

@socket_io.on("message")
def request(message):
    print("message : "+ message)
    to_client = dict()
    if message == 'new_connect':
        to_client['message'] = "메세지를 입력해주세요!!"
        to_client['type'] = 'connect'
    else:
        to_client['message'] = message
        to_client['type'] = 'normal'
    # emit("response", {'data': message['data'], 'username': session['username']}, broadcast=True)
    send(to_client, broadcast=True)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        list_numbers = []
        for n in range(5):
            numbers = gen_numbers_from_probability(ys_pred[0])
            numbers.sort()
            print('{0} : {1}'.format(n, numbers))
            list_numbers.append(numbers)

        return render_template('result.html', list_numbers=list_numbers)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True,port=8080)