from shutil import copy
from collections import defaultdict

# GPU 사용하기 위한 코드
from tensorflow.python.client import device_lib
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
print(device_lib.list_local_devices())
print("GPU Available: ", tf.test.is_gpu_available())

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 모델에 입력할 데이터를 준비하기 위한 코드:
# Training set 80%, Validation set 10%, Test set 10%로 나누어서 각 폴더에 이미지 복사하기
def prepare_data(filepath, src, dest):
    classes_images = defaultdict(list)

    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')

    count = 0  # 이미지 파일 개수를 카운트하는 변수
    for food in classes_images.keys():
        print("\nCopying images into", food)
        if not os.path.exists(os.path.join(dest, food)):
            os.makedirs(os.path.join(dest, food))

        str = dest[-10:]
        print(str)
        if str == 'validation':            # validation set(10%, 12625개의 이미지 파일)
            images = classes_images[food]  # 현재 음식 클래스에 대한 이미지 목록을 가져오기
            num_images_to_copy = len(images) // 2
            for i in range(num_images_to_copy):   # 0~124 (test.txt 파일에서)
                # src에서 dest 디렉터리로 이미지 복사하기
                copy(os.path.join(src, food, images[i]), os.path.join(dest, food, images[i]))
                count += 1                 # 이미지 파일을 복사할 때마다 개수를 증가시킴
        elif str == 'chive/test':          # test set(10%, 12625개의 이미지 파일)
            images = classes_images[food]  # 현재 음식 클래스에 대한 이미지 목록을 가져오기
            for i in range(125, len(images)):    # 125~250 (test.txt 파일에서)
                # src에서 dest 디렉터리로 이미지 복사하기
                copy(os.path.join(src, food, images[i]), os.path.join(dest, food, images[i]))
                count += 1                 # 이미지 파일을 복사할 때마다 개수를 증가시킴
        else:                              # train set(80%, 75750개의 이미지 파일)
            for i in classes_images[food]:
                copy(os.path.join(src, food, i), os.path.join(dest, food, i))
                count += 1                 # 이미지 파일을 복사할 때마다 개수를 증가시킴
    print("Copying Done!")
    return count                           # 이미지 파일 개수 반환

# prepare_data() 함수를 호출함으로써 train set, validation set, test set으로 데이터셋을 나누기
train_set_count = prepare_data('/home/npswml/Downloads/archive/meta/meta/train.txt', '/home/npswml/Downloads/archive/images',
             '/home/npswml/Downloads/archive/train')
valid_set_count = prepare_data('/home/npswml/Downloads/archive/meta/meta/test.txt', '/home/npswml/Downloads/archive/images',
             '/home/npswml/Downloads/archive/validation')
test_set_count = prepare_data('/home/npswml/Downloads/archive/meta/meta/test.txt', '/home/npswml/Downloads/archive/images',
             '/home/npswml/Downloads/archive/test')

# train, validation, test set 이미지 개수 출력
print("Number of images in train set:", train_set_count)  # 75750
print("Number of images in valid set:", valid_set_count)  # 12625
print("Number of images in test set:", test_set_count)    # 12625

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras.backend as K
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Add, AveragePooling2D, Flatten, Dense
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.optimizers import Adam
from PIL import Image
import tensorflow as tf

print(tf.__version__)
print(tf.test.gpu_device_name())

K.clear_session()

# 데이터 디렉토리 및 매개 변수 설정
train_data_dir = '/home/npswml/Downloads/archive/train'
validation_data_dir = '/home/npswml/Downloads/archive/validation'
test_data_dir = '/home/npswml/Downloads/archive/test'
img_width, img_height = 299, 299  # 각 음식 이미지의 크기: 299x299
nb_train_samples = 75750
nb_validation_samples = 12625
nb_test_samples = 12625
n_classes = 101
batch_size = 40
num_epochs = 50

# 데이터 생성기: 모델에 입력할 수 있는 형태로 데이터 생성
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# ResNet의 basic block 정의
def basic_block(input_tensor, filters, strides=1):
    # Convolutional layer 1
    # padding='same': 출력 이미지가 입력 이미지 크기와 동일함.
    x = Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Convolutional layer 2
    x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    if strides > 1:
        # If strides > 1, 1x1 convolution으로 downsampling 수행
        identity = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same', kernel_initializer='he_normal')(input_tensor)
        identity = BatchNormalization()(identity)
    else:
        # identity shortcut에 대한 identity 매핑하기
        identity = input_tensor

    # 기본 경로에 identity shortcut 추가
    x = Add()([x, identity])
    x = Activation('relu')(x)
    return x

# ResNet20 모델 아키텍처 정의
def resnet20(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)

    # Initial convolutional layer
    x = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks
    x = basic_block(x, filters=16)
    x = basic_block(x, filters=16)
    x = basic_block(x, filters=16)
    x = basic_block(x, filters=32, strides=2)
    x = basic_block(x, filters=32)
    x = basic_block(x, filters=32)
    x = basic_block(x, filters=64, strides=2)
    x = basic_block(x, filters=64)
    x = basic_block(x, filters=64)

    # Average pooling & classification(101개의 class로 분류하는) layer
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    x = Dense(num_classes, kernel_regularizer=l2(0.005), activation='softmax')(x)

    # model 생성
    model = Model(inputs=input_tensor, outputs=x)
    return model

# GPU 디바이스 지정
with tf.device('/device:GPU:0'):
    # ResNet20 model 생성
    input_shape = (img_width, img_height, 3)  # 입력 이미지의 형태 (가로: 299, 세로: 299, 채널: 3)
    num_classes = 101                         # 클래스 수
    model = resnet20(input_shape, num_classes)

    # 초기 학습 속도로 모델 컴파일
    learning_rate = 0.001
    decay_rate = learning_rate / num_epochs
    model.compile(optimizer=Adam(learning_rate=learning_rate, decay=decay_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # callbacks 정의
    checkpointer = ModelCheckpoint('resnet20.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    csv_logger = CSVLogger('resnet20_training.log')

    # model 훈련시키기 (Train)
    history = model.fit(train_generator,
                        steps_per_epoch = nb_train_samples // batch_size,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        callbacks=[csv_logger, checkpointer])
# 훈련한 모델 저장
model.save('model_trained.h5')

import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model

# 모든 음식 목록을 만들면서, 모든 음식 폴더가 있는 폴더의 경로를 넣기
def create_foodlist(path):
    list_ = list()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            list_.append(name)
    return list_

# 훈련하고 미세 조정한 모델 로드하기
my_model = load_model('model_trained.h5', compile=False)
food_list = create_foodlist("/home/npswml/Downloads/archive/images")

def predict_class(model, images, show=True):
    for img_path in images:
            # 이미지 load 및 전처리 과정
            img = tf.keras.utils.load_img(img_path, grayscale=False, color_mode='rgb', target_size=(img_width, img_height))

            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img /= 255.

            # 전처리된 이미지에 대한 예측 수행
            pred = model.predict(img)
            index = np.argmax(pred)

            # food_list 정렬
            food_list.sort()
            # 예측한 class label 가져오기
            pred_value = food_list[index]
            # 이미지 및 예측 클래스 레이블 표시
            if show:
                plt.imshow(img[0])
                plt.axis('off')
                plt.title(pred_value)
                plt.show()

# 예측할 이미지를 목록에 추가
images = []
images.append('Example1.jpg')
images.append('Example2.jpg')
images.append('Example3.jpg')
images.append('Example4.jpg')

print("PREDICTIONS BASED ON PICTURES UPLOADED")
predict_class(my_model, images, True)


# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# test dataset의 실제 레이블을 가져오기
test_labels = test_generator.classes

# test dataset에 대한 예측을 수행하기
test_predictions = model.predict(test_generator)
test_pred_labels = np.argmax(test_predictions, axis=1)

# confusion matrix 생성하기
cm = confusion_matrix(test_labels, test_pred_labels)

# confusion matrix 출력하기
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=food_list, yticklabels=food_list)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report

# 예측값과 실제값을 비교하여 classification_report를 생성
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

# classification_report를 출력하여 F1-score 확인
report = classification_report(y_true, y_pred)
print(report)

from sklearn.metrics import accuracy_score
# 예측값과 실제값을 비교하여 정확도 계산
accuracy = accuracy_score(y_true, y_pred)

# 정확도 출력
print("Accuracy:", accuracy)