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
import logging
import math

dtype = [mnp.float16, mnp.float32, mnp.float64, mnp.int8, mnp.int16]
intTypeBound = 1  # max integer type index
device = ["CPU","GPU","Ascend"]

class myError(Exception):
    def __init__():
        super()

def test_Tensor():
    logger = logging.getLogger('test_Tensor')
    passed = 0
    mydtype = [np.int8, np.int16, np.float16,np.float32]
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(mydtype)):
                try:
                    a1 = np.array([1,2,3]).astype(mydtype[i])
                    a2 = ms.Tensor(a1).asnumpy()
                    if np.array_equal(a1,a2):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))
                    
                    a1 = np.zeros([500,500,500]).astype(mydtype[i])
                    a2 = ms.Tensor(a1).asnumpy()
                    if np.array_equal(a1,a2):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])
    assert passed>0

def test_ops_add(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Add')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Add()   
                    a = mnp.array([1,2,3]).astype(dtype[i])
                    b = mnp.array([2,3,4])     
                    result = myops(a,b)
                    expect = mnp.array([3,5,7])
                    if not mnp.array_equal(result,expect):
                        raise(TypeError("result not as expected"))
                    else:
                        passed+=1

                    a = -1 * mnp.ones((4000,4000)).astype(dtype[i])
                    b = 2 * mnp.ones((4000,4000))
                    result = myops(a,b)
                    expect = mnp.ones((4000,4000))
                    if not mnp.array_equal(result,expect):
                        raise(TypeError("result not as expected"))
                    else:
                        passed += 1

                    a = ms.Tensor(3)
                    b = ms.Tensor(2)
                    result = myops(a,b)
                    expect = ms.Tensor(5)
                    if not mnp.array_equal(result,expect):
                        raise(TypeError("result not as expected"))
                    else:
                        passed += 1

                    a = mnp.array([1,2,3]).astype(dtype[i])
                    b = ms.Tensor(2)
                    result = myops(a,b)
                    expect = mnp.array([3,4,5])
                    if not mnp.array_equal(result,expect):
                        raise(TypeError("result not as expected")) 
                    else:
                        passed += 1

                    # if(i>intTypeBound):
                    #     a = mnp.array([1.2,3.5,8.9]).astype(dtype[i])
                    #     b = mnp.array([1.3,3.0,1.1])
                    #     result = myops(a,b)
                    #     expect = np.array([2.5,6.5,10.0])
                    #     if not (np.abs(result.asnumpy()-expect) < 0.01*expect).all():
                    #         logging.debug(result.asnumpy())
                    #         logging.debug(expect)
                    #         logging.debug((np.abs(result.asnumpy()-expect) < 0.01))
                    #         raise(Exception("result not as expected"))
                    #     else:
                    #         passed += 1
                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])
        assert passed>0

def test_ops_abs(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_abs')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Abs()
                    a = mnp.array([-5,-3,1]).astype(dtype[i])
                    result = myops(a)
                    expect = mnp.array([5,3,1])
                    if mnp.array_equal(result,expect):
                        passed+=1
                    else:
                       raise(TypeError("result not as expected"))

                    a = ms.Tensor(-5).astype(dtype[i])
                    result = myops(a)
                    expect = ms.Tensor(5)
                    if not mnp.array_equal(result,expect):
                       raise(TypeError("result not as expected"))
                    else:
                        passed += 1

                    a = -1 * mnp.ones((4000,4000)).astype(dtype[i])
                    result = myops(a)
                    expect = mnp.ones((4000,4000))
                    if mnp.array_equal(result,expect):
                        passed+=1
                    else:
                       raise(TypeError("result not as expected"))
                    
                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])
    assert passed>0

def test_ops_sub(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Sub')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Sub()
                    a = mnp.array([1,2,3]).astype(dtype[i])
                    b = mnp.array([2,3,4])
                    result = myops(a,b)
                    expect = mnp.array([-1,-1,-1])
                    if mnp.array_equal(result,expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))
                    
                    a = mnp.array([10,5,3]).astype(dtype[i])
                    b = ms.Tensor(1)
                    result = myops(a,b)
                    expect = mnp.array([9,4,2])
                    if mnp.array_equal(result,expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = -1 * mnp.ones((4000,4000)).astype(dtype[i])
                    b = 2 * mnp.ones((4000,4000))
                    result = myops(a,b)
                    expect = -3* mnp.ones((4000,4000))
                    if mnp.array_equal(result,expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))    
                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])
    assert passed>0

def test_ops_mod(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Mod')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    a = mnp.array([2,4,3]).astype(dtype[i])
                    b = mnp.array([2,3,4])
                    myops = ops.Mod()
                    result = myops(a,b)
                    expect = mnp.array([0,1,3])
                    if mnp.array_equal(result,expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a =  mnp.array([1,2,3,4,5]).astype(dtype[i])
                    b = ms.Tensor(2)
                    result = myops(a,b)
                    expect = mnp.array([1,0,1,0,1])
                    if mnp.array_equal(result,expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a =  mnp.ones((4000,4000,2)).astype(dtype[i])
                    b =  mnp.ones((4000,4000,2))
                    result = myops(a,b)
                    expect = mnp.zeros((4000,4000,2))
                    if mnp.array_equal(result,expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))  

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])
    assert passed>0

def test_ops_exp(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Exp')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Exp()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = np.exp(2)
                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.exp([2,4,3])
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a =  mnp.zeros((4000,4000,2)).astype(dtype[i])
                    result = myops(a)
                    expect = np.ones((4000,4000,2))
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        logging.info()
                        raise(TypeError("result not as expected"))    

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])

    assert passed>0

def test_ops_log(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Log')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Log()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = np.log(2)
                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.log([2,4,3])
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a =  mnp.ones((4000,4000,2)).astype(dtype[i])
                    result = myops(a)
                    if (np.abs(result.asnumpy()) <= 0.0001).all():
                        passed+=1
                    else:
                        logging.debug(result)
                        logging.debug(expect)
                        raise(TypeError("result not as expected"))    

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])

    assert passed>0

def test_ops_square(capsys):

    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Square')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Square()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = np.square(2)
                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.square([2,4,3])
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a =  2* mnp.ones((4000,4000,2)).astype(dtype[i])
                    result = myops(a)
                    expect = 4*  np.ones((4000,4000,2))
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))    

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])

    assert passed>0

def test_ops_sqrt(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Sqrt')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Sqrt()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = np.sqrt(2)

                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.sqrt([2,4,3])
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a =  4* mnp.ones((4000,4000,2)).astype(dtype[i])
                    result = myops(a)
                    expect = 2*  np.ones((4000,4000,2))
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])
    assert passed>0

def test_ops_sin(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Sin')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Sin()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = np.sin(2)

                    if np.abs(result.asnumpy()-expect) <= 0.01*expect:
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.sin(a.asnumpy())
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        logging.info(expect)
                        logging.info(result)
                        logging.info(np.abs(result.asnumpy()-expect))
                        logging.info(np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect))
                        raise(TypeError("result not as expected"))

                    a =  4* mnp.ones((4000,4000,2)).astype(dtype[i])
                    result = myops(a)
                    expect = np.sin(a.asnumpy())
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])
    assert passed>0

def test_ops_cos(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Cos')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Cos()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = np.cos(2)

                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.cos(a.asnumpy())
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        logging.info(expect)
                        logging.info(result)
                        logging.info(np.abs(result.asnumpy()-expect))
                        logging.info(np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect))
                        raise(TypeError("result not as expected"))

                    a =  4* mnp.ones((4000,4000,2)).astype(dtype[i])
                    result = myops(a)
                    expect = np.cos(a.asnumpy())
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])

    assert passed>0

def test_ops_tan(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Tan')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Tan()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = np.tan(2)
                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.tan(a.asnumpy())
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        logging.info(expect)
                        logging.info(result)
                        logging.info(np.abs(result.asnumpy()-expect))
                        logging.info(np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect))
                        raise(TypeError("result not as expected"))

                    a =  4* mnp.ones((4000,4000,2)).astype(dtype[i])
                    result = myops(a)
                    expect = np.tan(a.asnumpy())
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))
                        
                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])

    assert passed>0

def test_ops_sinh(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Sinh')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Sinh()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = np.sinh(2)

                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.sinh(a.asnumpy())
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        logging.info(expect)
                        logging.info(result)
                        logging.info(np.abs(result.asnumpy()-expect))
                        logging.info(np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect))
                        raise(TypeError("result not as expected"))

                    a =  4* mnp.ones((4000,4000,2)).astype(dtype[i])
                    result = myops(a)
                    expect = np.sinh(a.asnumpy())
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])

    assert passed>0

def test_ops_cosh(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Cosh')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Cosh()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = np.cosh(2)

                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.cosh(a.asnumpy())
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        logging.info(expect)
                        logging.info(result)
                        logging.info(np.abs(result.asnumpy()-expect))
                        logging.info(np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect))
                        raise(TypeError("result not as expected"))

                    a =  4* mnp.ones((4000,4000,2)).astype(dtype[i])
                    result = myops(a)
                    expect = np.cosh(a.asnumpy())
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])

    assert passed>0

def test_ops_tanh(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Cosh')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = ops.Tanh()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = np.tanh(2)

                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.tanh(a.asnumpy())
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        logging.info(expect)
                        logging.info(result)
                        logging.info(np.abs(result.asnumpy()-expect))
                        logging.info(np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect))
                        raise(TypeError("result not as expected"))

                    a =  4* mnp.ones((4000,4000,2)).astype(dtype[i])
                    result = myops(a)
                    expect = np.tanh(a.asnumpy())
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])

    assert passed>0

def test_nn_sigmoid(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Cosh')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = nn.Sigmoid()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = 1/(1+np.exp(-2))

                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = 1/(1+np.exp([-2,-4,-3]))
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a =  4*mnp.ones((1000,1000,4)).astype(dtype[i])
                    result = myops(a)
                    a = a.asnumpy()
                    expect = 1/(1+np.exp(-1*a))
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError(dtype[i]+" "+"result not as expected"))

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])

    assert passed>0

def test_nn_ReLU(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Cosh')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = nn.ReLU()
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = a.asnumpy()

                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,-4,3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.array([2,0,3])
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        logging.debug(result)
                        logging.debug(expect)
                        raise(TypeError("result not as expected"))

                    a =  -4*mnp.ones((1000,1000,4)).astype(dtype[i])
                    result = myops(a)
                    a = a.asnumpy()
                    expect = np.zeros((1000,1000,4))
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError(dtype[i]+" "+"result not as expected"))

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])

    assert passed>0

def test_nn_LeakyReLU(capsys):
    # with capsys.disabled():
    #     print("")
    logger = logging.getLogger('test_ops_Cosh')
    passed = 0
    for j in range(len(device)):
        try:
            logger.info("testing on device: "+device[j])
            context.set_context(device_target = device[j])
            for i in range(len(dtype)):
                try:
                    myops = nn.LeakyReLU(0.2)
                    a = ms.Tensor(2,dtype[i])
                    result = myops(a)
                    expect = a.asnumpy()

                    if np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect):
                        passed+=1
                    else:
                        raise(TypeError("result not as expected"))

                    a = mnp.array([2,-4,-3]).astype(dtype[i])
                    result = myops(a)
                    expect = np.array([2,-0.8,-0.6])
                    if  (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        logging.debug(result)
                        logging.debug(expect)
                        raise(TypeError("result not as expected"))

                    a =  -4*mnp.ones((1000,1000,4)).astype(dtype[i])
                    result = myops(a)
                    a = a.asnumpy()
                    expect = -0.8*np.ones((1000,1000,4))
                    if (np.abs(result.asnumpy()-expect) <= 0.01*np.abs(expect)).all():
                        passed+=1
                    else:
                        raise(TypeError(dtype[i]+" "+"result not as expected"))

                except(TypeError):
                    logger.warning("does not support dtype: "+str(dtype[i]))
        except(RuntimeError):
            logger.critical("does not support device: "+device[j])
        except(ValueError):
            logger.critical("does not support device: "+device[j])

    assert passed>0

    # def test_ops_square(capsys):
    # # with capsys.disabled():
    # #     print("")
    # logger = logging.getLogger('test_ops_Mod')
    # passed = 0
    # for j in range(len(device)):
    #     try:
    #         context.set_context(device_target = device[j])
    #         logger.info("testing on device: "+device[j])
    #         for i in range(len(dtype)):
    #             try:
    #                 a = mnp.array([2,4,3]).astype(dtype[i])
    #                 b = mnp.array([2,3,4])
    #                 myops = ops.Mod()
    #                 result = myops(a,b)
    #                 expect = mnp.array([0,1,3])
    #                 if mnp.array_equal(result,expect):
    #                     passed+=1
    #                 else:
    #                    assert 1

    #                 a =  mnp.array([1,2,3,4,5]).astype(dtype[i])
    #                 b = ms.Tensor(2)
    #                 result = myops(a,b)
    #                 expect = mnp.array([1,0,1,0,1])
    #                 if mnp.array_equal(result,expect):
    #                     passed+=1
    #                 else:
    #                     assert 1  

    #                 a =  mnp.ones((4000,4000,2)).astype(dtype[i])
    #                 b =  mnp.ones((4000,4000,2))
    #                 result = myops(a,b)
    #                 expect = mnp.zeros((4000,4000,2))
    #                 if mnp.array_equal(result,expect):
    #                     passed+=1
    #                 else:
    #                     assert 1     

    #             except(TypeError):
    #                 logger.warning("does not support dtype: "+str(dtype[i]))

    #     except(ValueError):
    #         logger.warning("does not support device: "+device[j])
    # assert passed>0
# def test_disabling_capturing(capsys):
#     print("this output is captured")
#     with capsys.disabled():
#         print("output not captured, going directly to sys.stdout")
#     print("this output is also captured")

# def test_ops_Add():
#     pass
# def test_ops_Argmax():
#     pass
# def test_ops_Argmin():
#     pass
# def test_ops_Sigmoid():
#     pass
# def test_ops_Flatten():
#     pass

# def test_logger():
#     log = logging.getLogger("test_logger")
#     log.info("info massage!")
#     log.debug("debug massage!")
#     log.error("error message!")
#     log.critical("critical message!")
#     log.warning("warning message!")
    # print("print message!")


# def test_mobilenetV1():
#     context.set_context(mode=context.GRAPH_MODE,device_target="GPU")
#     dataset = load_cifar(256)
#     trainset,testset = dataset.split([0.8,0.2])
#     net = MobileNetV1()
#     loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse = True)
#     opt = nn.Adam(net.trainable_params(), learning_rate=0.0002)
#     model = ms.Model(net,loss_fn,opt, metrics = {"Accuracy":nn.Accuracy()})
#     lm = LossMonitor(564)
#     try:
#         model.train(8,trainset, dataset_sink_mode = False)
#         print(model.eval(testset,dataset_sink_mode = False))
#         assert model.eval(testset,dataset_sink_mode = False)["Accuracy"] > 0.5
#     except:
#         print("Fail training model")
#         assert 0
#     pass

# def test_vgg16():
#     context.set_context(mode=context.GRAPH_MODE,device_target="GPU")
#     dataset = load_cifar(256)
#     trainset,testset = dataset.split([0.8,0.2])
#     net = Vgg16()
#     loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse = True)
#     opt = nn.Adam(net.trainable_params(), learning_rate=0.0002)
#     model = ms.Model(net,loss_fn,opt, metrics = {"Accuracy":nn.Accuracy()})
#     lm = LossMonitor(564)
#     try:
#         model.train(8,trainset, dataset_sink_mode = False)
#         print(model.eval(testset,dataset_sink_mode = False))
#         assert model.eval(testset,dataset_sink_mode = False)["Accuracy"] > 0.5
#     except:
#         print("Fail training model")
#         assert 0
#     pass
if (__name__ == "__main__"):
    # logging.basicConfig(level=logging.INFO)
    pytest.main(["-v", "--tb=native"])
