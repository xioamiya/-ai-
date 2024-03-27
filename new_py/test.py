#%%
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D

def create_conv_dense_model(input_shape, num_units, num_classes, activation='relu'):


    model = Sequential()

    # 添加第一个卷积层
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    # 添加第二个卷积层
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

    # 添加池化层
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 展平特征图
    model.add(Flatten())

    # 添加第一个全连接层
    model.add(Dense(num_units, activation=activation))

    # 添加第二个全连接层
    model.add(Dense(num_classes, activation='softmax'))

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 用法示例
input_shape = (64, 64, 3)  # 输入特征的形状
num_units = 64  # 隐藏层的神经元数量
num_classes = 3  # 输出类别的数量
model = create_conv_dense_model(input_shape, num_units, num_classes)

# 打印模型结构
model.summary()

# %%
from keras.models import Model
import matplotlib.pyplot as plt
import cv2
import numpy as np
# 创建一个新模型，该模型截取第一层卷积层的输出
conv_layer_name = 'conv2d_2'  # 替换为第一层卷积层的名称
conv_layer_model = Model(inputs=model.input, outputs=model.get_layer(conv_layer_name).output)

# 加载图像
img_path = '1.jpg'  # 替换为你的图像文件路径
img = cv2.imread(img_path)
img = cv2.resize(img, (64, 64))  # 调整大小
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
img = np.expand_dims(img, axis=0)  # 扩展维度，以匹配模型的输入形状
img = img / 255.0  # 预处理

# 获取第一层卷积层的输出
conv_layer_output = conv_layer_model.predict(img)

# 显示第一层卷积层的输出特征图
plt.figure(figsize=(8, 8))
for i in range(32):  # 32是假设第一层卷积层有32个滤波器
    plt.subplot(4, 8, i + 1)
    plt.imshow(conv_layer_output[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.show()

# %%
