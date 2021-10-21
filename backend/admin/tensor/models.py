import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from admin.common.models import ValueObject



class TensorFunction(object):
    def __init__(self):
        self.vo = ValueObject()
        self.vo.context = 'admin/tensor/data/'

    def hook(self):
        menu ='train_tf_model_by_random_data'
        if menu == 'tf_function':
            pass
        elif menu == 'tf_sum':
            result = self.tf_sum()
        elif menu == 'tf_add':
            result = self.tf_add()
        elif menu == 'create_model':
            self.create_model().summary()
        elif menu == 'create_tf_empty_model':
            self.create_tf_empty_model()
        elif menu == 'train_tf_model_by_random_data':
            self.train_tf_model_by_random_data()
        else:
            result = "해당사항 없음"
            print(f'결과: {result}')

    def train_tf_model_by_random_data(self):
        (x, y) = self.make_random_data()
        x_train, y_train = x[:150], y[:150]
        x_test, y_test = x[:150], y[:150]
        model = keras.models.load_model(f'{self.vo.context}simple_model.h5')
        history = model.fit(x_train, y_train, epochs=30, validation_split=0.3)
        epochs = np.arange(1, 30 + 1)
        plt.plot(epochs, history.history['loss'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{self.vo.context}simple_model.png')

    def create_tf_empty_model(self):
        '''
        model = keras.models.Sequential
        ([
            keras.layers.Flatten(input_shape=[1, 150]),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(units=1, activation="relu"),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(units=1, activation="softmax")
        ])
        model = Sequential()    # sequntial 모델 생성 할당 첫번째 층을
        model.add(Dense(32, input_shape=(16, ))) # 첫번째 층을 dense 32 크기 out
        model.add(Dense(32))
        Arguments:
        units: 현재 dense 를 통해서 만들 hidden layer 의 Node 의 수
        첫번째 인자 : 출력 뉴런의 수를 설정합니다.
        input_dim : 입력 뉴런의 수를 설정합니다.
        init : 가중치 초기화 방법 설정합니다.
        uniform : 균일 분포
        normal : 가우시안 분포
        activation : 활성화 함수 설정합니다.
        linear : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.
        relu : rectifier 함수, 은익층에 주로 쓰입니다.
        sigmoid : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다.
        softmax : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다.
        다중클래스 분류문제에서는 클래스 수만큼 출력 뉴런이 필요합니다.
        만약 세가지 종류로 분류한다면, 아래 코드처럼 출력 뉴런이 3개이고,
        입력 뉴런과 가중치를 계산한 값을 각 클래스의 확률 개념으로 표현할 수 있는
        활성화 함수인 softmax를 사용합니다.
        '''
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=1, activation='relu', input_dim=1))
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.Dense(units=1, activation='softmax'))
        model.compile(optimizer='sgd', loss='mse')
        model.save(f'{self.vo.context}simple_model.h5')


    def make_random_data(self):
        x = np.random.uniform(low=-2, high=2, size=200)
        y = []
        for t in x:
            r = np.random.normal(loc=0.0, scale=(0.5 + t * t /3), size=None)
            y.append(r)
        return x, 1.726*x - 0.84 + np.array(y)

    def create_model(self) -> object:
        input = tf.keras.Input(shape=(1,))
        output = tf.keras.layers.Dense(1)(input)
        model = tf.keras.Model(input, output)
        return model

    @tf.function
    def tf_sum(self):
        a = tf.constant(1, tf.float32)
        b = tf.constant(2, tf.float32)
        c = tf.constant(3, tf.float32)
        z = a + b + c
        return z

    @tf.function
    def tf_add(self):
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        z = tf.add(x, y)
        # z = tf.subtract(x, y)
        # z = tf.multiply(x, y)
        # z = tf.divide(x, y)
        return z



    def tf_function(self):
        mnist = tf.keras.datasets.mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0
        X_train = X_train[..., tf.newaxis]
        X_test = X_test[..., tf.newaxis]
        train_ds = tf.data.Dataset.from_tensors_slices(
            (X_train, y_train)
        ).shuffle(10000).batch(32)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
        return list(train_ds.as_numpy_iterator())
        '''
        train_ds : <class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>
        '''
        # print(list(train_ds.as_numpy_iterator()))
        '''
        plt.figure(figsize=(10, 10))
        plt.grid(False)
        plt.imshow(train_ds[3])
        plt.savefig(f'{self.vo.context}train_ds.png')
        plt.imshow(test_ds[3])
        plt.savefig(f'{self.vo.context}test_ds.png')
        '''


class FashionClassification(object):
    def __init__(self):
        self.vo = ValueObject()
        self.vo.context = 'admin/tensor/data/'
        self.class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def fashion(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        # self.peek_datas(train_images, test_images, test_labels)
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=[28, 28]),
            keras.layers.Dense(128, activation="relu"),  # neron count 128
            keras.layers.Dense(10, activation="softmax")  # 출력층 활성화함수는 softmax
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=5)
        model.save(f'{self.vo.context}fashion_classification.h5')


    def peek_datas(self, train_images, test_images, train_labels):
        print(train_images.shape)
        print(test_images.dtype)
        print(f'훈련행: {train_images.shape[0]} 열: {train_images.shape[1]}')
        print(f'테스트행: {test_images.shape[0]} 열: {test_images.shape[1]}')
        plt.figure()
        plt.imshow(train_images[3])
        plt.colorbar()
        plt.grid(False)
        plt.savefig(f'{self.vo.context}fashion_random.png')
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(train_images[i], cmap=plt.cm.binary)
            plt.xlabel(self.class_name[train_labels[i]])
        plt.savefig(f'{self.vo.context}fashion_subplot.png')


    def test_and_save_images(self, model, test_images, test_labels):
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        predictions = model.predict(test_images)
        i = 5
        print(f'모델이 예측한 값 {np.argmax(predictions[i])}')
        print(f'정답: {test_labels[i]}')
        print(f'테스트 정확도: {test_acc}')
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        test_image, test_predictions, test_label = test_images[i], predictions[i], test_labels[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(test_image, cmap=plt.cm.binary)
        test_pred = np.argmax(test_predictions)

        if test_pred == test_label:
            color = 'blue'
        else:
            color = 'red'
        plt.xlabel('{} : {} %'.format(self.class_name[test_pred],
                                      100 * np.max(test_predictions),
                                      self.class_name[test_label], color))
        plt.subplot(1, 2, 2)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        this_plot = plt.bar(range(10), test_pred, color='#777777')
        plt.ylim([0,1])
        test_pred = np.argmax(test_predictions)
        this_plot[test_pred].set_color('red')
        this_plot[test_label].set_color('blue')
        plt.savefig(f'{self.vo.context}fashion_answer2.png')





    '''

    def train_model(self, model, train_images, train_labels) -> object:
        model.fil(train_images, train_labels, epoch=5)
        return model

    def test_model(self, model, test_images, test_labels) -> object:
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print(f'테스트 정확도: {test_acc}')

    def predict(self, model, test_images, test_labels, index):
        prediction = model.predict(test_images)
        pred = prediction[index]
        answer = test_labels[index]
        print(f'모델이 예측한 값 {np.argmax(pred)}')
        print(f'정답: {answer}')
        return [prediction, test_images, test_labels]

    def plot_image(self):
        pass

    def plot_value_array(self):
        pass
    '''



class AdalineGD(object): # 적응형 선형 뉴런 분류기

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        # X : {array-like}, shape = [n_samples, n_features]
        #           n_samples 개의 샘플과 n_features 개의 특성으로 이루어진 훈련 데이터

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = [] # 에포크마다 누적된 비용 함수의 제곱합

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,
            # in the case of logistic regression (as we will see later),
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X): # 단위 계단 함수를 사용하여 클래스 레이블을 반환
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


class Calculator(object):

    def __init__(self):
        print(f'Tensorflow Version: {tf.__version__}')

    def process(self):
        self.plus(4, 8)
        print('*'*100)
        self.mean()


    def plus(self, a, b):
        print(tf.constant(a) + tf.constant(b))


    def mean(self):
        x_array = np.arange(18).reshape(3,2,3)
        x2 = tf.reshape(x_array, shape=(-1, 6))
        #  각 열 합 계산
        xsum = tf.reduce_sum(x2, axis=0)
        #  각 일 평균 계산
        xmean = tf.reduce_mean(x2, axis=0)

        print(f'입력 크기: {x_array.shape} \n')
        print(f'크기가 변경된 입력 크기: {x2.numpy()}\n')
        print(f'열의 합: {xsum.numpy()}\n')
        print(f'열의 평균: {xmean.numpy()}\n')


