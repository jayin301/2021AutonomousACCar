# PyTorch Tutorial

현재 폴더의 tutorial 코드는 아래 링크의 튜토리얼, 각종 youtube 영상을 참고하여 작성되었습니다.
* youtube [Sung Kim] https://www.youtube.com/watch?v=TxIVr-nk1so
* youtube [테디노트] https://www.youtube.com/watch?v=1Q_etC_GHHk 
* youtube [김군이] https://www.youtube.com/watch?v=E0R9Xf_GyUc
* youtube [딥러닝호형] https://www.youtube.com/watch?v=8PnxJ3s3Cwo
* 파이토치 한국 사용자 모임 튜토리얼 https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html 
-----------------------
## Deep Learning Basic

### Supervised Learning
* regression
* binary classification
* multi-label classification

### linear regression의 hypothesis와 cost function
![image](https://user-images.githubusercontent.com/62630731/125215017-f7124780-e2f4-11eb-9112-f490cc4bda74.png)
즉, cost function은 weight와 bias에 대한 함수이며 **cost를 최소화하는 weight와 bias를 구하는 것**이 목표

> #### 1. 경사하강법 Gradient Descent Method
> 어떤 점에서 시작하든지 미분을 통해 경사도를 계산해서 최저점을 구하는 것
> #### 2. 오차역전파 BackPropagation
> 인공신경망(artificial neural network)을 학습시키기 위한 알고리즘으로 간단히 말하면, 원하는 target의 값과 모델이 계산해낸 값의 차이를 구한 후 그 오차 값을 뒤로 전파해나가면서 각 node가 가지는 weight 값을 갱신하는 과정
> 
> 순방향 계산(forward pass)시에는 local gradient를 미리 구해서 저장할 수 있고, global gradient는 계산된 값을 역으로 전달받아야 그 값을 구할 수 있다. 이때, chain rule에 의거해서 **local gradient * global gradient = d(loss)/dx**
------------------------
## PyTorch Basic

파이토치(PyTorch)의 목적
- Numpy 연산을 GPU의 파워로 가속화
- 딥러닝의 연산을 더 유연하고 빠르게

### 1. 파이토치 연산 기본 단위, 텐서(Tensor)
* 단일 데이터 타입으로 된 자료들의 다차원 행렬
* CPU로 연산을 수행할 지 또는 GPU로 연산을 수행할 지 선택할 수 있다

### 2. Dataset & DataLoader
- `Dataset`: 샘플과 정답(label)을 저장
- `DataLoader`: 샘플에 쉽게 접근하기 위해 Dataset을 반복 가능한 객체(iterable)로 wrapping

#### Transfrom
파이토치 제공 데이터의 이미지(feature) 기본 형태는 `PIL Image`, 정답(label)은 정수형이므로 데이터를 불러올 때 **학습에 적합하도록 전처리** 필요 (`torchvision.transfroms`에서 제공)

> * **transform** = feature normalization
> * ToTensor() = PIL Image나 numpy ndarray를 floatTensor로 변환
> ```python
> import torchvision.transfroms as tr
> tr.Compose([tr.Resize(8), tr.ToTensor()]) # Compose = 순서대로 실행
> ```
>
> * **target_transform** = label transform 
> * Lamda 함수 적용 = integer to one-hot encoded signed tensor

#### Dataset 구조의 이해
```python
type(training_data)	    >> <class 'torchvision.datasets.mnist.FashionMNIST'>
len(training_data)          >> 60000
type(training_data[0])      >> <class ‘tuple’>
train_features.size()       >> torch.Size([64, 1, 28, 28]) # [batch, channel, image size]
```
> 1. image
> ```python
> type(training_data[0][0])	>> <class ‘torch.Tensor’>
> training_data[0][0].shape	>> torch.Size([1, 28, 28]) # grayscale이므로 단일 채널
> ```
> 2. label: 0부터 9까지 라벨링된 10가지 종류의 의류

#### Autograd
`autograd`를 통해 backpropagation을 위한 미분값을 자동으로 계산
> `autograd.Variable`의 구성
> 1. `data` = Tensor 형태의 데이터
> 2. `grad` = data가 거쳐온 layer에 대한 미분값 축적
> 3. `grad_fn` = 어떤 연산(함수)에 대한 미분값을 계산했는지

