import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorcircuit as tc
import tensorflow as tf


K = tc.set_backend("tensorflow")
# 512维的结果
n = 9
blocks = 5


# circuit9
def qml(x, weights):
    c = tc.Circuit(n, inputs=x)
    # tf.print(c.state())
    # print(c.state().numpy())
    for j in range(blocks):
        for i in range(n):
            c.H(i)
        for i in range(n - 1):            
            c.cry(i, i + 1, theta=weights[j, i, 1])

    # 添加一层新的线路
        for i in range(n):
            c.rx(i, theta=weights[j, i, 0])
            
    outputs = K.stack(
        [K.real(c.expectation([tc.gates.z(), [i]])) for i in range(n)]
    )
    # print(f'output in qml is {outputs}')
    outputs = K.reshape(outputs, [-1]) # 还需要添加一个全连接层 将其输出为2
    return outputs


qml_vmap = K.vmap(qml, vectorized_argnums=0)
qml_layer = tc.interfaces.torch_interface(qml_vmap, jit=True)


class Quantum(nn.Module):
    def __init__(self):
        super(Quantum, self).__init__()
        self.q_weights = nn.Parameter(torch.randn([blocks, n, 2]))
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(5, 1), stride=(1,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(3, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(5, 1), stride=(1,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(5, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=8, kernel_size=(5, 1), stride=(2,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(8, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 1), stride=(1,1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 1), stride=(1, 1), padding=0),
            nn.ReLU(True),
            nn.BatchNorm2d(16, affine=True, track_running_stats=True, eps=1e-5, momentum=0.1),
            # nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0),
        )

        # self.fc1 = nn.Linear(1520, 512)
        self.fc2 = nn.Linear(9, 2)

        
    def forward(self, input):

        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        output = conv5_output.view(-1, 512) # 这个地方的数值是不能进行随意更改的
        # fc_output = self.fc1(conv5_output)
        # # print(fc_output.shape)
        # # 注意传入的数据需要进行归一化
        # # fc_norm = (conv5_output - torch.mean(conv5_output, dim=0)) / torch.std(conv5_output, dim=0)
        # # print(f'fc_norm is {fc_norm}')
        q_output = qml_layer(output, self.q_weights)
        
        # # print(f'q_output is {q_output}')
        out = self.fc2(q_output)
        # out = self.fc(conv5_output)

        return out