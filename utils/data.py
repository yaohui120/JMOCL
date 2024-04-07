import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import os
import pdb

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]

    class_order = np.arange(100).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(100).tolist()
        self.class_order = class_order
        self.image_size = [32,32,3]

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(self.args["data_path"], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(self.args["data_path"], train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class iCIFAR100_224(iCIFAR100):
    train_trsf = [
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/imagenet100/train/'
        test_dir = 'data/imagenet100/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    def __init__(self, args):
        self.args = args
        class_order = np.arange(1000).tolist()
        self.class_order = class_order
        self.image_size = [224,224,3]

    def download_data(self):
        # train_dir = '/mnt/Data/myh/Datasets/imagenet-r/train/'
        # test_dir = '/mnt/Data/myh/Datasets/imagenet-r/val/'

        # train_dset = datasets.ImageFolder(train_dir)
        # test_dset = datasets.ImageFolder(test_dir)

        # self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        # self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        rootdir = self.args["data_path"]

        train_txt = rootdir+'/train.txt'
        test_txt = rootdir+'/test.txt'

        train_images, train_labels = [], []
        with open(train_txt, 'r') as dict_file:
            for line in dict_file:
                (key, value) = line.strip().split('\t')
                train_images.append(os.path.join(rootdir, value))
                train_labels.append(int(key))
        train_images, train_labels = np.array(train_images), np.array(train_labels)

        test_images, test_labels = [], []
        with open(test_txt, 'r') as dict_file:
            for line in dict_file:
                (key, value) = line.strip().split('\t')
                test_images.append(os.path.join(rootdir, value))
                test_labels.append(int(key))
        test_images, test_labels = np.array(test_images), np.array(test_labels)

        self.train_data, self.train_targets = train_images, train_labels
        self.test_data, self.test_targets = test_images, test_labels

class iCUB200_224(iData):
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/cub_200/train/'
        test_dir = 'data/cub_200/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iCARS196_224(iData):
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/cars196/train/'
        test_dir = 'data/cars196/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iResisc45_224(iData): 
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/resisc45/train/'
        test_dir = 'data/resisc45/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iSketch345_224(iData):
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/sketch345/train/'
        test_dir = 'data/sketch345/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        
class iCore50(iData):
    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    class_order = np.arange(8 * 50).tolist()

    def __init__(self, args):
        self.args = args
        self.image_size = [128,128,3]

    def download_data(self):
        from utils.datautils.core50.core50data import CORE50
        datagen = CORE50(root=self.args["data_path"], scenario="ni")

        dataset_list = []
        for i, train_batch in enumerate(datagen):
            imglist, labellist = train_batch
            labellist += i*50
            imglist = imglist.astype(np.uint8)
            dataset_list.append([imglist, labellist])
        train_x = np.concatenate(np.array(dataset_list, dtype=object)[:, 0])
        train_y = np.concatenate(np.array(dataset_list, dtype=object)[:, 1])

        self.train_data = train_x
        self.train_targets = train_y

        test_x, test_y = datagen.get_test_set()
        test_x = test_x.astype(np.uint8)
        self.test_data = test_x
        self.test_targets = test_y

class i5Datasets_vit(iData):
    use_path = False
    train_trsf = [
        transforms.Resize(224),
        # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = np.arange(50).tolist()
        self.class_order = class_order
        self.image_size = [64,64,3]

    def download_data(self):
        from PIL import Image
        img_size=64
        train_dataset = datasets.cifar.CIFAR10(self.args["data_path"], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(self.args["data_path"], train=False, download=True)

        trainlist = []
        testlist = []
        train_label_list = []
        test_label_list = []

        # cifar10
        cifar10_train_dataset = datasets.cifar.CIFAR10(self.args["data_path"], train=True, download=True)
        cifar10_test_dataset = datasets.cifar.CIFAR10(self.args["data_path"], train=False, download=True)
        for img, target in zip(cifar10_train_dataset.data, cifar10_train_dataset.targets):
            trainlist.append(np.array(Image.fromarray(img).resize((img_size, img_size))))
            train_label_list.append(target)
        for img, target in zip(cifar10_test_dataset.data, cifar10_test_dataset.targets):
            testlist.append(np.array(Image.fromarray(img).resize((img_size, img_size))))
            test_label_list.append(target)

        # MNIST
        minist_train_dataset = datasets.MNIST(self.args["data_path"], train=True, download=True)
        minist_test_dataset = datasets.MNIST(self.args["data_path"], train=False, download=True)
        for img, target in zip(minist_train_dataset.data.numpy(), minist_train_dataset.targets.numpy()):
            trainlist.append(np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')))
            train_label_list.append(target+10)
        for img, target in zip(minist_test_dataset.data.numpy(), minist_test_dataset.targets.numpy()):
            testlist.append(np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')))
            test_label_list.append(target+10)

        # notMNIST
        classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        tarin_dir = self.args["data_path"]
        test_dir = self.args["data_path"]
        for idx, cls in enumerate(classes):
            image_files = os.listdir(os.path.join(tarin_dir,"notMNIST_large", cls))
            for img_path in image_files:
                try:
                    image = np.array(Image.open(os.path.join(tarin_dir,"notMNIST_large", cls, img_path)).resize((img_size, img_size)).convert('RGB'))
                    trainlist.append(image)
                    train_label_list.append(idx+20)
                except:
                    print(os.path.join(tarin_dir,"notMNIST_large", cls, img_path))
            image_files = os.listdir(os.path.join(test_dir,"notMNIST_small", cls))
            for img_path in image_files:
                try:
                    image = np.array(Image.open(os.path.join(test_dir,"notMNIST_small", cls, img_path)).resize((img_size, img_size)).convert('RGB'))
                    testlist.append(image)
                    test_label_list.append(idx+20)
                except:
                    print(os.path.join(test_dir,"notMNIST_small", cls, img_path))

        # Fashion-MNIST
        fminist_train_dataset = datasets.FashionMNIST(self.args["data_path"], train=True, download=True)
        fminist_test_dataset = datasets.FashionMNIST(self.args["data_path"], train=False, download=True)
        for img, target in zip(fminist_train_dataset.data.numpy(), fminist_train_dataset.targets.numpy()):
            trainlist.append(np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')))
            train_label_list.append(target+30)
        for img, target in zip(fminist_test_dataset.data.numpy(), fminist_test_dataset.targets.numpy()):
            testlist.append(np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')))
            test_label_list.append(target+30)

        # SVHN
        svhn_train_dataset = datasets.SVHN(self.args["data_path"], split='train', download=True)
        svhn_test_dataset = datasets.SVHN(self.args["data_path"], split='test', download=True)
        for img, target in zip(svhn_train_dataset.data, svhn_train_dataset.labels):
            trainlist.append(np.array(Image.fromarray(img.transpose(1,2,0)).resize((img_size, img_size))))
            train_label_list.append(target+40)
        for img, target in zip(svhn_test_dataset.data, svhn_test_dataset.labels):
            testlist.append(np.array(Image.fromarray(img.transpose(1,2,0)).resize((img_size, img_size))))
            test_label_list.append(target+40)

        train_dataset.data = np.array(trainlist)
        train_dataset.targets = np.array(train_label_list)
        test_dataset.data = np.array(testlist)
        test_dataset.targets = np.array(test_label_list)
        # pdb.set_trace()
        # train_num = []
        # ttt = np.array(train_dataset.targets)
        # for i in range(50):
        #     train_num.append(ttt[ttt==i].shape[0])
        # train:(772371, 64, 64, 3); test:(74756, 64, 64, 3)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class CDDB(object):
    use_path = True
    def __init__(self, args):
        self.args = args
        class_order = args.class_order
        self.class_order = class_order

    def download_data(self):
        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args.task_name):
            root_ = os.path.join(self.args.data_path, name, 'train')
            sub_classes = os.listdir(root_) if self.args.multiclass[id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        for id, name in enumerate(self.args.task_name):
            root_ = os.path.join(self.args.data_path, name, 'val')
            sub_classes = os.listdir(root_) if self.args.multiclass[id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)

class iDomainNet(iData):
    # for domain incremental learning, need to change
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    
    def __init__(self, args):
        self.args = args
        class_order = np.arange(1200).tolist()
        self.class_order = class_order
        self.image_size = [224,224,3]
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]

    def download_data(self):
        self.image_list_root = self.args["data_path"]

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "train" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.train_data = np.array(train_x)
        self.train_targets = np.array(train_y)

        image_list_paths = [os.path.join(self.image_list_root, d + "_" + "test" + ".txt") for d in self.domain_names]
        imgs = []
        for taskid, image_list_path in enumerate(image_list_paths):
            image_list = open(image_list_path).readlines()
            imgs += [(val.split()[0], int(val.split()[1]) + taskid * 345) for val in image_list]
        train_x, train_y = [], []
        for item in imgs:
            train_x.append(os.path.join(self.image_list_root, item[0]))
            train_y.append(item[1])
        self.test_data = np.array(train_x)
        self.test_targets = np.array(train_y)
        print('Loaded DomainNet!')
        print('Handling...')
        num = []
        for i in np.unique(self.train_targets):
            if i > 344:
                num[i%345] += self.train_targets[self.train_targets==i].shape[0]
            else:
                num.append(self.train_targets[self.train_targets==i].shape[0])
        num = np.array(num)
        class_need = np.argsort(num)[-200:] # 取所有域上训练样本最多的前200类
        class_need = np.sort(class_need)
        cld = np.unique(self.train_targets)
        for i in np.unique(self.train_targets):
            if i%345 not in class_need:
                ind = np.where(cld == i)
                cld = np.delete(cld, ind)
        # 2:1划分训练集测试集
        train_ind = self.train_targets<345*4
        test_ind = self.test_targets>=345*4
        self.train_data, self.train_targets = self.train_data[train_ind], self.train_targets[train_ind]
        self.test_data, self.test_targets = self.test_data[test_ind], self.test_targets[test_ind]
        mapping = {}
        jj = 0
        for i in cld:
            mapping[i] = jj
            jj += 1
        for i in np.unique(self.train_targets):
            if i not in class_need:
                ind = np.where(self.train_targets%345 == i)
                self.train_data = np.delete(self.train_data, ind)
                self.train_targets = np.delete(self.train_targets, ind)
                ind = np.where(self.test_targets%345 == i)
                self.test_data = np.delete(self.test_data, ind)
                self.test_targets = np.delete(self.test_targets, ind)
        for i in cld:
            self.train_targets[self.train_targets==i] = mapping[i]
            self.test_targets[self.test_targets==i] = mapping[i]
            self.test_targets = self.test_targets % 345
