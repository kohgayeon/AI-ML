# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans

# Libraries added without justification are a minus factor.


# Seed setting
seed_num = 2022
# np.random.seed(seed_num)
iteration = 100  # Number of times to repeat steps E and M.


class EM:
    """ expectation-maximization algorithm, EM algorithm
    The EM class is a class that implements an EM algorithm using GMM and kmeans.

    Within the fit function, the remaining functions should be used.
    Other functions can be added, but all tasks must be implemented with Python built-in functions and Numpy functions.
    You should annotate each function with a description of the function and parameters(Leave a comment).
    """

    def __init__(self, n_clusters, iteration):
        """
        Parameters
        ----------
        n_clusters (int): Num of clusters (num of GMM)
        iteration (int): Num of iteration
            Termination conditions if the model does not converge
        mean (ndarray): Num of clusters x Num of features
            The mean vector that each cluster has.
        sigma (ndarray): Num of clusters x Num of features x Num of features
            The covariance matrix that each cluster has.
        pi (ndarray): Num of labels (num of clusters)
            z(x), Prior probability that each cluster has.
        return None.
        -------
        None.

        """
        self.n_clusters = n_clusters
        self.iteration = iteration
        self.mean = np.zeros((3, 4))       # 3x4 크기의 2차원 array -> 평균
        self.sigma = np.zeros((3, 4, 4))   # 3x4x4 크기의 3차원 array -> 표준편차 (covariance matrix)
        self.pi = np.zeros(3)              # 3개 lable의 확률을 갖는 1차원 array -> prior probability

    def initialization(self, data):
        """1.initialization, 10 points
        Initial values for mean, sigma, and pi should be assigned.
        It have a significant impact on performance.
        """
        # 랜덤 데이터 점을 사용하여 self.mean 초기화
        random_point = np.random.choice(len(data), self.n_clusters, replace=False)
        self.mean = data[random_point]

        # self.pi를 균일하게 초기화
        self.pi.fill(1 / self.n_clusters)

        # 작은 랜덤 값으로 self.sigma 초기화
        for k in range(self.n_clusters):
            self.sigma[k] = np.eye(4) * 0.1  # 4x4 matrix -> 대각 행렬

        return

    def multivariate_gaussian_distribution(self, data, mean, sigma):
        """ 2.multivariate_gaussian_distribution, 10 points
        Use the linear algebraic functions of Numpy. π of this function is not self.pi

        your comment here
        """
        # multivariate_gaussian_distribution (다변량 가우시안 분포)의 pdf 계산식
        new_data = data - mean   # data: 1x4 matrix, mean: 1x4
        pdf = (1. / (np.sqrt((2 * np.pi) ** self.n_clusters * np.linalg.det(sigma))) * np.exp(-(np.linalg.solve(sigma, new_data).T.dot(new_data)) / 2))
        return pdf

    def expectation(self, data):
        """ 3.expectation step, 20 points
        The multivariate_gaussian_distribution(MVN) function must be used.

        your comment here
        """
        posterior = np.zeros((len(data), self.n_clusters))  # 반환할 posterior은 150x3 matrix

        for n in range(len(data)):
            posterior_denom = 0                             # posterior를 계산하기 위한 분모 정의
            for j in range(self.n_clusters):                # 다변량 가우시안 분포를 사용해서 분모 posterior_denom 계산
                posterior_denom += self.pi[j] * EM_model.multivariate_gaussian_distribution(data[n], self.mean[j], self.sigma[j])
            for k in range(self.n_clusters):                # 다변량 가우시안 분포를 사용해서 (posterior)의 분자 계산
                pdf = EM_model.multivariate_gaussian_distribution(data[n], self.mean[k], self.sigma[k])
                posterior[n][k] = pdf * self.pi[k] / posterior_denom

        return posterior  # 150x3 크기의 matrix 반환

    def maximization(self, posterior, data):
        """ 4.maximization step, 20 points
        Hint. np.outer

        your comment here
        """
        mean = np.zeros((3, 4))
        sigma = np.zeros((3, 4, 4))
        pi = np.zeros(3)
        for k in range(self.n_clusters):  # cluster 개수(class 개수=3)만큼 반복
            mean_exp = 0                  # mean 분자 정의
            sigma_exp = 0                 # sigma 분자 정의
            for n in range(len(data)):    # 데이터 개수(N=150)만큼 반복 -> mean 분자 & sigma 분자 계산
                mean_exp += posterior[n][k] * data[n]
                sigma_exp += posterior[n][k] * np.outer(data[n] - self.mean[k], data[n] - self.mean[k])
            pi[k] = np.sum(posterior[:, k]) / len(data)     # class별 pi 업데이트: 3x1 크기
            mean[k] = mean_exp / np.sum(posterior[:, k])    # class별 mean 업데이트: 3x4 크기
            sigma[k] = sigma_exp / np.sum(posterior[:, k])  # class별 sigma 업데이트: 3x4x4 크기

        return mean, sigma, pi

    def fit(self, data):
        """ 5.fit clustering, 20 points
        Functions initialization, expectation, and maximization should be used by default.
        Termination Condition. Iteration is finished or posterior is the same as before. (Beware of shallow copy)
        Prediction for return should be formatted. Refer to iris['target'] format.

        your comment here
        """
        posterior = np.zeros((150, 3)) # expectation 단계를 통해 계산된 posterior
        prev = np.zeros((150, 3))      # 이전 iteration의 posterior를 저장할 변수 선언
        EM_model.initialization(data)  # mean, sigma, pi 변수 초기화 함수 호출
        tolerance = 1e-6               # 수렴 허용 오차

        for t in range(iteration):
            # expectation 단계를 통해서 posterior update
            posterior = EM_model.expectation(data)
            # maximization 단계를 통해서 파라미터 (mean, sigma, pi) estimation
            estimation_mean, estimation_sigma, estimation_pi = EM_model.maximization(posterior, data)
            # termination condition -> mean, sigma, pi, posterior가 모두 수렴하는지 검사
            # 수렴 허용 오차(tolerance)보다 이전의 값(self.mean)과 추정한 값(estimation_mean)의 차이가 더 작은 경우 수렴한다고 간주함.
            if t > 0:
                if np.max(abs(estimation_mean - self.mean)) < tolerance and np.max(abs(estimation_sigma - self.sigma)) < tolerance and np.max(abs(estimation_pi - self.pi)) < tolerance:
                    if np.max(abs(posterior - prev)) < tolerance:
                        break
            # estimation_mean, estimation_sigma, estimation_pi, posterior 값을 이전의 값에 복사해서 update
            self.mean = estimation_mean.copy()
            self.sigma = estimation_sigma.copy()
            self.pi = estimation_pi.copy()
            prev = posterior

        # MLE(Maximum Likelihood Estimation)를 적용해 prediction 값을 추정함.
        prediction = np.argmax(posterior, axis=1)  # np array (150) as assigned by labels 0, 1, 2
        return prediction

def plotting(data):
    """ 6.plotting, 20 points with report
    Default = seaborn pairplot

    your comment here
    """
    # seaborn pairplot() 호출 -> 각 column별 데이터에 대한 상관관계나 분류적 특성을 보여주는 그래프 출력
    sns.pairplot(data, hue="labels", markers=["o", "s", "D"])
    plt.show()
    return

if __name__ == '__main__':
    # Loading and labeling data
    iris = datasets.load_iris()
    original_data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['labels'])
    original_data['labels'] = original_data['labels'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    plotting(original_data)

    # Only data is used W/O labels beacause EM and Kmeans are unsupervised learning
    data=iris['data']

    # Unsupervised learning(clustering) using EM algorithm
    EM_model = EM(n_clusters=3, iteration=iteration)
    EM_pred = EM_model.fit(data)
    EM_pd = pd.DataFrame(data=np.c_[data, EM_pred], columns=iris['feature_names'] + ['labels'])
    plotting(EM_pd)

    # Why are these two elements almost the same? Write down the reason in your report. Additional 10 points
    print(f'pi :            {EM_model.pi}')
    print(f'count / total : {np.bincount(EM_pred) / 150}')

    # Unsupervised learning(clustering) using KMeans algorithm
    KM_model = KMeans(n_clusters=3, init='random', random_state=seed_num, max_iter=iteration).fit(data)
    KM_pred = KM_model.predict(data)
    KM_pd = pd.DataFrame(data=np.c_[data, KM_pred], columns=iris['feature_names'] + ['labels'])
    plotting(KM_pd)

    # No need to explain.
    for idx in range(2):
        EM_point = np.argmax(np.bincount(EM_pred[idx * 50:(idx + 1) * 50]))
        KM_point = np.argmax(np.bincount(KM_pred[idx * 50:(idx + 1) * 50]))
        EM_pred = np.where(EM_pred == idx, 3, EM_pred)
        EM_pred = np.where(EM_pred == EM_point, idx, EM_pred)
        EM_pred = np.where(EM_pred == 3, EM_point, EM_pred)
        KM_pred = np.where(KM_pred == idx, 3, KM_pred)
        KM_pred = np.where(KM_pred == KM_point, idx, KM_pred)
        KM_pred = np.where(KM_pred == 3, KM_point, KM_pred)

    EM_hit = np.sum(iris['target'] == EM_pred)
    KM_hit = np.sum(iris['target'] == KM_pred)
    print(f'EM Accuracy: {round(EM_hit / 150, 2)}    Hit: {EM_hit} / 150')
    print(f'KM Accuracy: {round(KM_hit / 150, 2)}    Hit: {KM_hit} / 150')