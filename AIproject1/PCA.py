#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

print("people.images.shape : {}".format(people.images.shape))
print("클래스 개수 : {}".format(len(people.target_names)))


# In[6]:


# 각 타깃이 나타난 횟수 
counts = np.bincount(people.target)

# 타깃별 이름과 횟수 출력
for i,(count,name) in enumerate(zip(counts,people.target_names)):

    print("{0:25} {1:3}".format(name,count))

    if (i+1) %3 ==0 :

        print()


# In[11]:


import matplotlib.pyplot as plt
#from joblib import Memory
import mglearn

mglearn.plots.plot_pca_whitening()

plt.show()


# In[14]:


# KNN 모델을 학습시키는 과정 (PCA 화이트닝 X)
# 정확도: 0.140

from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

# 각 target이 나타난 횟수 계산 
counts=np.bincount(people.target) # people.target의 빈도 계산


mask = np.zeros(people.target.shape, dtype=bool)
for target in np.unique(people.target):             # 중복을 제거한 target 리스트에서 한 개의 원소 선택

    mask[np.where(people.target==target)[0][:50]]=1 # 데이터 편중을 막기 위해 50개의 이미지만 선택

x_people = people.data[mask]   # 훈련 데이터 생성
y_people = people.target[mask] # 테스트 데이터 생성 

# 전처리 메소드 import
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
x_people_scaled = scaler.fit_transform(x_people) # 전처리 메소드 적용


# 테스트/훈련세트 나누기 (전처리한 데이터 분할)
x_train, x_test, y_train, y_test = train_test_split(x_people_scaled, y_people,         # 분할할 데이터
                                                    stratify=y_people, random_state=0) # 그룹화할 데이터, 랜덤상태

knn = KNeighborsClassifier(n_neighbors=1) # 이웃의 수
knn.fit(x_train, y_train)                 # 모델 학습
print("1-최근접 이웃 테스트 정확도 : {:.3f}".format(knn.score(x_test, y_test)))


# In[10]:


# KNN 모델을 학습시키는 과정 (PCA 화이트닝 O) -> PCA를 하여 고유 얼굴 성분 출력함.
# 정확도: 0.159

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

# PCA 모델 생성 및 적용
pca = PCA(n_components=100, whiten=True, random_state = 0) # 주성분 갯수, whitening option, 랜덤상태
pca.fit(x_train) # PCA 학습

x_train_pca = pca.transform(x_train) # PCA를 데이터에 적용
x_test_pca = pca.transform(x_test)


# PCA를 적용한 데이터 형태
print('x_train_pca.shape \ntrain형태:{}'.format(x_train_pca.shape)) # (1547, 100)
print('x_test_pca.shape \ntest형태:{}'.format(x_test_pca.shape))    # (516, 100)


# 머신 러닝 모델 생성 및 학습
knn = KNeighborsClassifier(n_neighbors=1) # 이웃의 수
knn.fit(x_train_pca, y_train) # 모델 학습
print('1-최근접 이웃 테스트 정확도(pca 화이트닝 옵션): {:.3f}'.format(knn.score(x_test_pca, y_test)))


#모델의 정확도가 14%에서 약16%로 상승


#이미지 데이터일 경우엔 계산한 주성분을 쉽게 시각화 가능

#주성분이 나타내는 것은 입력 데이터 공간에서의 어떤 방향임을 기억

#입력 차원이 87 x 65픽셀의 흑백 이미지이고 따라서 주성분또한 87 x 65


print('pca.components_.shape \n{}'.format(pca.components_.shape)) # (100, 5655)


# subplot 3x5를 axes에 할당
fig, axes = plt.subplots(3, 5, figsize=(15, 12),
                         subplot_kw={'xticks':(), 'yticks':()}) # subplot 축 설정

# pca.components_와 axes.ravel()을 하나씩 순서대로 할당한 후 인덱스 부여
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):

    ax.imshow(component.reshape(image_shape), cmap='viridis') # image_shape= (87, 65)

    ax.set_title("principle component {}".format(i + 1))                # image title

plt.show()


# In[19]:


# 주성분의 개수를 다르게 하여 얼굴 이미지 재구성하기

import mglearn
mglearn.plots.plot_pca_faces(x_train, x_test, image_shape) # 훈련데이터, 테스트데이터, 이미지크기(87x65)

plt.show() # 그림 출력

