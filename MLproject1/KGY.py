import numpy as np

def feature_normalization(data):  # 10 points
    # parameter 
    feature_num = data.shape[1]  # data의 열의 개수 = 4
    data_point = data.shape[0]   # data의 행의 개수 = 150

    # you should get this parameter correctly
    normal_feature = np.zeros([data_point, feature_num])  # 150*4
    mu = np.zeros([feature_num])                          # 4
    std = np.zeros([feature_num])                         # 4

    # your code here
    mu = np.mean(data, 0)  # 각 feature의 평균 mean 계산하기
    std = np.std(data, 0)  # 각 feature의 표준편차 standard deviation 계산하기

    for i in range(len(data)):         # 세로 길이(150행)
        for j in range(len(data[i])):  # 가로 길이(4열)
            normal_feature[i][j] = ((data[i][j] - mu[j]) / std[j]) - 1  # Normalization data
    # end

    return normal_feature


def split_data(data, label, split_factor):  # train data과 test data로 나눠주는 함수 (split_factor = 100)
    # 순서대로 train data(처음부터 100까지 슬라이싱), test data(100부터 마지막까지 슬라이싱),
    # train label(처음부터 100까지 슬라이싱), test label(100부터 마지막까지 슬라이싱)을 반환함.
    return data[:split_factor], data[split_factor:], label[:split_factor], label[split_factor:]


def get_normal_parameter(data, label, label_num):  # 20 points
    # parameter
    feature_num = data.shape[1]

    # you should get this parameter correctly    
    mu = np.zeros([label_num, feature_num])      # 3x4 mean matrix, data는 2차원 배열, label은 1차원
    sigma = np.zeros([label_num, feature_num])

    # your code here
    se_0array = []  # 각 label과 feature에 해당하는 list 선언(Setosa, feature0)
    se_1array = []  # (Setosa, feature1)
    se_2array = []  # (Setosa, feature2)
    se_3array = []  # (Setosa, feature3)

    ve_0array = []  # (Versicolor, feature0)
    ve_1array = []  # (Versicolor, feature1)
    ve_2array = []  # (Versicolor, feature2)
    ve_3array = []  # (Versicolor, feature3)

    vi_0array = []  # (Virginica, feature0)
    vi_1array = []  # (Virginica, feature1)
    vi_2array = []  # (Virginica, feature2)
    vi_3array = []  # (Virginica, feature3)

    for i in range(len(data)):         # 위에서 선언한 각 리스트에 해당하는 label과 feature에 맞춰서 저장해두기
        for j in range(len(data[i])):  # 4개의 feature (0, 1, 2, 3)
            if j == 0:  # feature0
                if label[i] == 0:                 # Setosa
                    se_0array.append(data[i][j])
                elif label[i] == 1:               # Versicolor
                    ve_0array.append(data[i][j])
                elif label[i] == 2:               # Virginica
                    vi_0array.append(data[i][j])
            elif j == 1:  # feature1
                if label[i] == 0:
                    se_1array.append(data[i][j])
                elif label[i] == 1:
                    ve_1array.append(data[i][j])
                elif label[i] == 2:
                    vi_1array.append(data[i][j])
            elif j == 2:  # feature2
                if label[i] == 0:
                    se_2array.append(data[i][j])
                elif label[i] == 1:
                    ve_2array.append(data[i][j])
                elif label[i] == 2:
                    vi_2array.append(data[i][j])
            else:  # feature3
                if label[i] == 0:
                    se_3array.append(data[i][j])
                elif label[i] == 1:
                    ve_3array.append(data[i][j])
                elif label[i] == 2:
                    vi_3array.append(data[i][j])

    # 각 리스트에 저장한 값들을 가지고 해당하는 label과 feature의 순서대로 평균과 표준편차를 계산 후,
    # mu와 sigma 3*4 배열을 채우기
    mu[0][0] = np.mean(se_0array)    # (Setosa, feature0) 리스트의 평균 계산
    sigma[0][0] = np.std(se_0array)  # (Setosa, feature0) 리스트의 표준편차 계산
    mu[0][1] = np.mean(se_1array)
    sigma[0][1] = np.std(se_1array)
    mu[0][2] = np.mean(se_2array)
    sigma[0][2] = np.std(se_2array)
    mu[0][3] = np.mean(se_3array)
    sigma[0][3] = np.std(se_3array)

    mu[1][0] = np.mean(ve_0array)
    sigma[1][0] = np.std(ve_0array)
    mu[1][1] = np.mean(ve_1array)
    sigma[1][1] = np.std(ve_1array)
    mu[1][2] = np.mean(ve_2array)
    sigma[1][2] = np.std(ve_2array)
    mu[1][3] = np.mean(ve_3array)
    sigma[1][3] = np.std(ve_3array)

    mu[2][0] = np.mean(vi_0array)
    sigma[2][0] = np.std(vi_0array)
    mu[2][1] = np.mean(vi_1array)
    sigma[2][1] = np.std(vi_1array)
    mu[2][2] = np.mean(vi_2array)
    sigma[2][2] = np.std(vi_2array)
    mu[2][3] = np.mean(vi_3array)
    sigma[2][3] = np.std(vi_3array)

    # end

    return mu, sigma


def get_prior_probability(label, label_num):  # 10 points
    # parameter
    data_point = label.shape[0]  # 전체 label의 개수

    # you should get this parameter correctly
    prior = np.zeros([label_num])

    # your code here
    cnt_0 = 0   # label0(Seotosa)인 개수를 세기 위한 변수
    cnt_1 = 0   # label1(Versicolor)인 개수를 세기 위한 변수
    cnt_2 = 0   # label2(Virginica)인 개수를 세기 위한 변수
    for i in label:
        if i == 0:
            cnt_0 = cnt_0 + 1
        elif i == 1:
            cnt_1 = cnt_1 + 1
        else:
            cnt_2 = cnt_2 + 1

    prior[0] = cnt_0 / data_point  # 전체 label에서 label0가 나타날 확률
    prior[1] = cnt_1 / data_point  # 전체 label에서 label1가 나타날 확률
    prior[2] = cnt_2 / data_point  # 전체 label에서 label2가 나타날 확률

    # end
    return prior


def Gaussian_PDF(x, mu, sigma):  # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    pdf = 0

    # your code here
    # 가우시안 정규분포 식 구현하기 (데이터 값 x, 평균 mu, 표준편차 sigma)
    pdf = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
    # end

    return pdf


def Gaussian_Log_PDF(x, mu, sigma):  # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    log_pdf = 0

    # your code here
    # 가우시안 정규분포 식에서 로그 취해서 구현하기 (데이터 값 x, 평균 mu, 표준편차 sigma)
    log_pdf = np.log(1 / np.sqrt(2 * np.pi * sigma ** 2)) - (- (x - mu) ** 2 / (2 * sigma ** 2))
    # end

    return log_pdf


def Gaussian_NB(mu, sigma, prior, data):  # 40 points
    # parameter
    data_point = data.shape[0]
    label_num = mu.shape[0]

    # you should get this parameter correctly   
    likelihood = np.zeros([data_point, label_num])      # 50*3 크기의 2차원 배열
    posterior = np.zeros([data_point, label_num])       # 50*3 크기의 2차원 배열
    ## evidence can be ommitted because it is a constant

    # your code here
    ## Function Gaussian_PDF or Gaussian_Log_PDF should be used in this section
    # likelihood: [0][0]은 class0이라고 가정하고 data의 첫 번째 행에 있는 4개의 feature들로 가우시안 함수 호출을 통해 계산해서 넣기

    for k in range(0, 3):                     # class: 0, 1, 2
        for i in range(len(data)):
            feature = 1
            for j in range(len(data[i])):     # 4번 옆으로 이동하면서 data의 한 행에 있는 4개 feature 값을 불러옴.
                # Gaussian_PDF 함수를 호출해서 likelihood 계산함. -> 이때 4개의 값을 누적해서 곱한다.
                # 만약 Gaussian_Log_PDF 함수를 호출해서 계산한다면, feature = 0으로 초기화하고 누적해서 더하면 된다.
                feature = Gaussian_PDF(data[i][j], mu[k][j], sigma[k][j]) * feature
            likelihood[i][k] = feature                    # 각 class에 해당하는 likelihood 저장하기
            posterior[i][k] = np.log(feature * prior[k])  # np.log()를 통해 자연로그 취하기 -> posterior 추정하기

    # end
    return posterior


def classifier(posterior):  # posterior를 사용해 classifier를 정의함
    data_point = posterior.shape[0]
    prediction = np.zeros([data_point])

    # MLE(Maximum Likelihood Estimation)를 적용해 prediction 값을 추정함.
    prediction = np.argmax(posterior, axis=1)

    return prediction


def accuracy(pred, gnd):           # 파라미터: prediction, test_label
    data_point = len(gnd)

    hit_num = np.sum(pred == gnd)  # 예측한 값과 정답 값이 맞은 경우를 모두 더해서 최종적으로 정답을 맞춘 개수를 구함.

    return (hit_num / data_point) * 100, hit_num  # 정답을 맞춘 확률(정확도; accuracy)과 정답을 맞춘 개수를 반환함.

    ## total 100 point you can get