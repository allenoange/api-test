import pytest
import mindspore as ms
import numpy as np
import mindspore.ops as ops
import mindspore.numpy as mnp
import mindspore.context as context
import time
import mindspore.nn as nn
from load_dataset import load_cifar
from model.vgg16 import Vgg16
from model.mobilenetv1 import  MobileNetV1
from mindspore.train.callback import  LossMonitor 

def test_Tensor1():
    # log = logging.getLogger('test_Tensor1')
    a1 = np.array([1,2,3])
    start = time.time()
    a2 = ms.Tensor(a1).asnumpy()
    end = time.time()
    mytime = str(end-start)
    print("running time: ",mytime,"s")
    # log.info(mytime)
    assert np.array_equal(a1, a2)
def test_Tensor2():
    a1 = np.ones((10000,10000))
    start = time.time()
    a2 = ms.Tensor(a1).asnumpy()
    end = time.time()
    mytime = str(end-start)
    print("running time: ",mytime,"s")
    assert np.array_equal(a1, a2)

def test_Tensor_add():
    a1 = mnp.array([1,2,3])
    a2 = mnp.array([4,5,6])
    b = a1+a2
    assert ms.numpy.array_equal(b,mnp.array([5,7,9]))

def test_ops_Abs():
    a1 = mnp.array(-5).astype(mnp.float32)
    myops = ops.Abs()
    b = myops(a1)
    assert ms.numpy.array_equal(b,mnp.array(5))

def test_ops_Add():
    pass
def test_ops_Argmax():
    pass
def test_ops_Argmin():
    pass
def test_ops_Sigmoid():
    pass
def test_ops_Flatten():
    pass

def test_mobilenetV1():
    context.set_context(mode=context.GRAPH_MODE,device_target="GPU")
    dataset = load_cifar(256)
    trainset,testset = dataset.split([0.8,0.2])
    net = MobileNetV1()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse = True)
    opt = nn.Adam(net.trainable_params(), learning_rate=0.0002)
    model = ms.Model(net,loss_fn,opt, metrics = {"Accuracy":nn.Accuracy()})
    lm = LossMonitor(564)
    try:
        model.train(8,trainset, dataset_sink_mode = False)
        print(model.eval(testset,dataset_sink_mode = False))
        assert model.eval(testset,dataset_sink_mode = False)["Accuracy"] > 0.5
    except:
        print("Fail training model")
        assert 0
    pass

def test_vgg16():
    context.set_context(mode=context.GRAPH_MODE,device_target="GPU")
    dataset = load_cifar(256)
    trainset,testset = dataset.split([0.8,0.2])
    net = Vgg16()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse = True)
    opt = nn.Adam(net.trainable_params(), learning_rate=0.0002)
    model = ms.Model(net,loss_fn,opt, metrics = {"Accuracy":nn.Accuracy()})
    lm = LossMonitor(564)
    try:
        model.train(8,trainset, dataset_sink_mode = False)
        print(model.eval(testset,dataset_sink_mode = False))
        assert model.eval(testset,dataset_sink_mode = False)["Accuracy"] > 0.5
    except:
        print("Fail training model")
        assert 0
#     pass
if (__name__ == "__main__"):
    # logging.basicConfig(level=logging.INFO)
    pytest.main(['-v', '--capture=no'])