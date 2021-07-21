import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype
def load_cifar(batch_size = 32):
    path = "data/cifar-10-batches-bin"
    dataset = ds.Cifar10Dataset(dataset_dir=path, shuffle=False)
    scale = 1/255.0
    shift = 0.0
    normalize = CV.Normalize(( 0.485, 0.456, 0.406 ), ( 0.229, 0.224, 0.225 ))
    rescale = CV.Rescale(scale,shift)
    hwc2chw = CV.HWC2CHW()
    hflip = CV.RandomHorizontalFlip(0.2)
    vflip = CV.RandomVerticalFlip(0.2)
    randomrotate = CV.RandomRotation(5)
    ops = [rescale, normalize, hflip,vflip, randomrotate, hwc2chw]
    typecast = C.TypeCast(mstype.int32)
    dataset = dataset.map(operations = ops, input_columns = ["image"])
    dataset = dataset.map(operations = typecast, input_columns = ["label"])
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(3)
    return dataset

if __name__ =="__main__":
    import matplotlib.pyplot as plt
    data = load_cifar()
    for i in data.create_dict_iterator():
        image = i["image"].asnumpy()
        # print(image.shape)
        plt.imshow(image[0].transpose([1,2,0]))
        plt.show()
        break
