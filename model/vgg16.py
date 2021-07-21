import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp

class Vgg16(nn.Cell):
    def __init__(self, img_size = 32, classnum=10):
        super(Vgg16,self).__init__()
        if(not img_size%32==0):
            raise(Exception("input size must be multiple of 32"))
        self.size = int(img_size/32)
        self.relu = nn.ReLU()
        self.Conv2d_1 = nn.Conv2dBnAct(3,64,3,1,activation = 'relu',weight_init="XavierUniform")
        self.Conv2d_2 = nn.Conv2dBnAct(64,64,3,1,activation = 'relu' ,weight_init="XavierUniform")
        self.Conv2d_3 = nn.Conv2dBnAct(64,128,3,1,activation = 'relu' ,weight_init="XavierUniform")
        self.Conv2d_4 = nn.Conv2dBnAct(128,128,3,1,activation = 'relu' ,weight_init="XavierUniform")
        self.Conv2d_5 = nn.Conv2dBnAct(128,256,3,1,activation = 'relu' ,weight_init="XavierUniform")
        self.Conv2d_6 = nn.Conv2dBnAct(256,256,3,1,activation = 'relu' ,weight_init="XavierUniform")
        self.Conv2d_7 = nn.Conv2dBnAct(256,512,3,1,activation = 'relu',weight_init="XavierUniform")
        self.Conv2d_8 = nn.Conv2dBnAct(512,512,3,1,activation = 'relu',weight_init="XavierUniform")
        self.maxpool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Dense(self.size*self.size*512,4096)
        self.fc2 = nn.Dense(4096,4096)
        self.fc3 = nn.Dense(4096,classnum)
        self.dropout = nn.Dropout(0.25)

    def construct(self,x):
        x = self.Conv2d_1(x)
        x = self.maxpool(self.relu(self.Conv2d_2(x)))
        # x=self.dropout(x)

        x = self.Conv2d_3(x)
        x = self.maxpool(self.relu(self.Conv2d_4(x)))
        # x=self.dropout(x)

        x = self.Conv2d_5(x)
        x = self.Conv2d_6(x)
        x = self.maxpool(self.relu(self.Conv2d_6(x)))
        # x=self.dropout(x)

        x = self.Conv2d_7(x)
        x = self.Conv2d_8(x)
        x = self.maxpool(self.relu(self.Conv2d_8(x)))
        # x=self.dropout(x)

        x = self.Conv2d_8(x)
        x = self.Conv2d_8(x)
        x = self.maxpool(self.relu(self.Conv2d_8(x)))
        # x=self.dropout(x)

        x = x.view(-1,self.size*self.size*512)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return x
    
if __name__ == "__main__":
    net = Vgg16(32)
    my_input = mnp.zeros((16,3,32,32))
    print(net(my_input).shape)