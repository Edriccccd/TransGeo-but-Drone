import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from PIL import Image
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import scipy.io as sio


class LimitedFoV(object):
    def __init__(self, fov=360.):
        self.fov = fov

    def __call__(self, x):
        angle = random.randint(0, 359)
        rotate_index = int(angle / 360. * x.shape[2])
        fov_index = int(self.fov / 360. * x.shape[2])
        if rotate_index > 0:
            img_shift = torch.zeros(x.shape)
            img_shift[:,:, :rotate_index] = x[:,:, -rotate_index:]
            img_shift[:,:, rotate_index:] = x[:,:, :(x.shape[2] - rotate_index)]
        else:
            img_shift = x
        return img_shift[:,:,:fov_index]


def input_transform_fov(size, fov):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        LimitedFoV(fov=fov),
    ])

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# pytorch implementation of CVACT loader
class Uni_1652(torch.utils.data.Dataset):
    def __init__(self, mode = '', root = 'D:/DATA_set/University-Release', same_area=True, print_bool=False,args=None,names=['satellite','drone']):
        super(Uni_1652, self).__init__()
        
        self.args = args
        self.root = root
        self.mode = mode
        self.names = names
        self.sat_size = [512, 512]  # [512, 512]
        self.sat_size_default = [512, 512]  # [512, 512]
        self.drone_size = [512, 512]  # [224, 1232]
        if args.sat_res != 0:
            self.sat_size = [args.sat_res, args.sat_res]

        if print_bool:
            print(self.sat_size, self.drone_size)

        self.sat_ori_size = [512, 512]
        self.drone_ori_size = [512, 512]

        # if args.fov != 0:
        #     self.transform_query = input_transform_fov(size=self.grd_size,fov=args.fov)
        # else:
        self.transform_query = input_transform(size=self.drone_size) #Drone

        # if len(polar) == 0:
        self.transform_reference = input_transform(size=self.sat_size) #Sati
        # else:
        #     self.transform_reference = input_transform(size=[750,750])

        self.to_tensor = transforms.ToTensor()
        dict_path = {}
        for name in names:
            dict_ = {}
            for cls_name in os.listdir(os.path.join(root, 'train', name)):
                img_list = os.listdir(os.path.join(root, 'train', name, cls_name))
                img_path_list = [os.path.join(root,'train',name,cls_name,img) for img in img_list]
                dict_[cls_name] = img_path_list
            dict_path[name] = dict_
        cls_names = os.listdir(os.path.join(root, 'train', names[0]))
        cls_names.sort()
        map_dict={i:cls_names[i] for i in range(len(cls_names))}
        
        
        
        self.cls_names = cls_names
        self.map_dict = map_dict
        self.dict_path = dict_path
        self.index_cls_nums = 2
        
        
        dict_test_path = {}
        test_names = ['satellite','drone']
        for test_name in test_names:
            dict_test_ = {}
            for cls_test_name in os.listdir(os.path.join(root, 'train', test_name)):
                img_list = os.listdir(os.path.join(root, 'train', test_name, cls_test_name))
                img_path_list = [os.path.join(root,'train',test_name,cls_test_name,img) for img in img_list]
                dict_test_[cls_test_name] = img_path_list
            dict_test_path[test_name] = dict_test_

        cls_names = os.listdir(os.path.join(root, 'train', test_names[0]))
        cls_names.sort()
        map_test_dict={i:cls_names[i] for i in range(len(cls_names))}

        
        self.cls_test_names = cls_names
        self.map_test_dict = map_test_dict
        self.dict_test_path = dict_test_path
        self.index_test_cls_nums = 2
            
        # dict_test_path = {}
        # test_names = ['query_satellite', 'gallery_drone']
        # for test_name in test_names:
        #     dict_test_ = {}
        #     for cls_test_name in os.listdir(os.path.join(root, 'test', test_name)):
        #         img_list = os.listdir(os.path.join(root, 'test', test_name, cls_test_name))
        #         img_path_list = [os.path.join(root,'test',test_name,cls_test_name,img) for img in img_list]
        #         dict_test_[cls_test_name] = img_path_list
        #     dict_test_path[test_name] = dict_test_

        # cls_names = os.listdir(os.path.join(root, 'test', test_names[0]))
        # cls_names.sort()
        # map_test_dict={i:cls_names[i] for i in range(len(cls_names))}
        
        # self.cls_test_names = cls_names
        # self.map_test_dict = map_test_dict
        # self.dict_test_path = dict_test_path
        # self.index_test_cls_nums = 2
            
        # anuData = sio.loadmat(os.path.join(self.root, 'ACT_data.mat'))

        # self.id_all_list = []
        # self.id_idx_all_list = []
        # idx = 0
        # missing = 0
        # for i in range(0, len(anuData['panoIds'])):
        #     grd_id = 'streetview/' + anuData['panoIds'][i] + '_grdView.jpg'
        #     sat_id = 'satview_polish/' + anuData['panoIds'][i] + '_satView_polish.jpg'

        #     self.id_all_list.append([sat_id, grd_id])
        #     self.id_idx_all_list.append(idx)
        #     idx += 1

        if print_bool:
            print('Uni_1652: load',' data_size =', len(self.map_dict))


    def sample_from_cls(self,name,cls_num):
        img_path = self.dict_path[name][cls_num]
        img_path = np.random.choice(img_path,1)[0]
        img = Image.open(img_path).convert('RGB')
        return img
    
    def sample_from_cls_test(self,name_test,cls_test_names):
        test_img_path = self.dict_test_path[name_test][cls_test_names]
        test_img_path = np.random.choice(test_img_path,1)[0]
        test_img = Image.open(test_img_path).convert('RGB')
        return test_img
    
    # def sample_from_cls_test(self,name_test,cls_test_names):
    #     test_img_path = self.dict_test_path[name_test][cls_test_names]
    #     test_img_path = np.random.choice(test_img_path,1)[0]
    #     test_img = Image.open(test_img_path).convert('RGB')
    #     return test_img
    
    def __getitem__(self, index):
        if self.mode== 'train' :
            idx = index % len(self)
            cls_nums = self.map_dict[index]
            img = self.sample_from_cls("satellite",cls_nums)
            img_s = self.transform_reference(img)
            img = self.sample_from_cls("drone",cls_nums)
            img_d = self.transform_query(img)
            if self.args.crop:
                atten_sat = Image.open(os.path.join('D:/Trans_drone/Save/attention','train',str(idx)+'.png')).convert('RGB')
                return img_s, img_d, torch.tensor(idx), torch.tensor(idx), 0, self.to_tensor(atten_sat)
            return img_s,img_d,torch.tensor(idx),torch.tensor(idx),0,0
        
        # elif 'scan_val' in self.mode:
        #     cls_nums = self.map_test_dict[index]
        #     img = self.sample_from_cls_test("query_satellite",cls_nums)
        #     img_s = self.transform_reference(img)
        #     img = self.sample_from_cls_test("gallery_drone",cls_nums)
        #     img_q = self.transform_query(img)
        #     return img_s, img_q, torch.tensor(index), torch.tensor(index), 0, 0
        
        elif 'scan_val' in self.mode:
            cls_nums = self.map_test_dict[index]
            img = self.sample_from_cls_test("satellite",cls_nums)
            img_s = self.transform_reference(img)
            img = self.sample_from_cls_test("drone",cls_nums)
            img_q = self.transform_query(img)
            return img_s, img_q, torch.tensor(index), torch.tensor(index), 0, 0
        
        # elif 'test_reference' in self.mode:
        #     cls_nums = self.map_test_dict[index]
        #     img = self.sample_from_cls_test("query_satellite",cls_nums)
        #     img_s = self.transform_reference(img)
        #     if self.args.crop:
        #         atten_sat = Image.open(os.path.join('D:/Trans_drone/Save/attention','val',str(index)+'.png')).convert('RGB')
        #         return img_s, torch.tensor(index), self.to_tensor(atten_sat)
        #     return img_s, torch.tensor(index), 0
        
        elif 'test_reference' in self.mode:
            cls_nums = self.map_test_dict[index]
            img = self.sample_from_cls("satellite",cls_nums)
            img_s = self.transform_reference(img)
            if self.args.crop:
                atten_sat = Image.open(os.path.join('D:/Trans_drone/Save/attention','val',str(index)+'.png')).convert('RGB')
                return img_s, torch.tensor(index), self.to_tensor(atten_sat)
            return img_s, torch.tensor(index), 0
        
        # elif 'test_query' in self.mode:
        #     cls_nums = self.map_test_dict[index]
        #     img = self.sample_from_cls_test("gallery_drone",cls_nums)
        #     img_q = self.transform_query(img)
        #     return img_q, torch.tensor(index), torch.tensor(index)
        
        elif 'test_query' in self.mode:
            cls_nums = self.map_test_dict[index]
            img = self.sample_from_cls("drone",cls_nums)
            img_q = self.transform_query(img)
            return img_q, torch.tensor(index), torch.tensor(index)
        
        else:
            print('not implemented!!!!')
            raise Exception
            


    def __len__(self):
        # return len(self.cls_names)
        if self.mode == 'train':
            return len(self.cls_names)
        elif 'scan_val' in self.mode:
            return len(self.cls_test_names)
        elif 'test_reference' in self.mode:
            return len(self.cls_test_names)
        elif 'test_query' in self.mode:
            return len(self.cls_test_names)
        else:
            print('not implemented!')
            raise Exception
        # self.id_list = []
        # self.id_idx_list = []
        # self.training_inds = anuData['trainSet']['trainInd'][0][0] - 1
        # self.trainNum = len(self.training_inds)
        # if print_bool:
        #     print('CVACT train:', self.trainNum)

        # for k in range(self.trainNum):
        #     sat_id = self.id_all_list[self.training_inds[k][0]][0]
        #     grd_id = self.id_all_list[self.training_inds[k][0]][1]
        #     if not os.path.exists(os.path.join(self.root, grd_id)) or not os.path.exists(os.path.join(self.root, sat_id)):
        #         if print_bool:
        #             print('train:',k, grd_id, sat_id)
        #         missing += 1
        #     else:
        #         self.id_list.append(self.id_all_list[self.training_inds[k][0]])
        #         self.id_idx_list.append(k)

        # self.val_inds = anuData['valSet']['valInd'][0][0] - 1
        # self.valNum = len(self.val_inds)
        # if print_bool:
        #     print('CVACT val:', self.valNum)

        # self.id_test_list = []
        # self.id_test_idx_list = []
        # for k in range(self.valNum):
        #     sat_id = self.id_all_list[self.val_inds[k][0]][0]
        #     grd_id = self.id_all_list[self.val_inds[k][0]][1]
        #     if not os.path.exists(os.path.join(self.root, grd_id)) or not os.path.exists(
        #             os.path.join(self.root, sat_id)):
        #         if print_bool:
        #             print('val', k, grd_id, sat_id)
        #         missing += 1
        #     else:
        #         self.id_test_list.append(self.id_all_list[self.val_inds[k][0]])
        #         self.id_test_idx_list.append(k)
        # if print_bool:
        #     print('missing:', missing)  # may miss some images

    # def __getitem__(self, index, debug=False):
    #     if self.mode== 'train':
    #         idx = index % len(self.id_idx_list)
    #         img_query = Image.open(self.root + self.id_list[idx][1]).convert('RGB')
    #         img_query = img_query.crop((0,img_query.size[1]//4,img_query.size[0],img_query.size[1]//4*3))
    #         img_reference = Image.open(self.root + self.id_list[idx][0]).convert('RGB')
    #         img_query = self.transform_query(img_query)
    #         img_reference = self.transform_reference(img_reference)
    #         if self.args.crop:
    #             atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','train',str(idx)+'.png')).convert('RGB')
    #             return img_query, img_reference, torch.tensor(idx), torch.tensor(idx), 0, self.to_tensor(atten_sat)
    #         return img_query, img_reference, torch.tensor(idx), torch.tensor(idx), 0, 0

    #     elif 'scan_val' in self.mode:
    #         img_reference = Image.open(self.root + self.id_test_list[index][0]).convert('RGB')
    #         img_reference = self.transform_reference(img_reference)
    #         img_query = Image.open(self.root + self.id_test_list[index][1]).convert('RGB')
    #         img_query = img_query.crop((0, img_query.size[1] // 4, img_query.size[0], img_query.size[1] // 4 * 3))
    #         img_query = self.transform_query(img_query)
    #         return img_query, img_reference, torch.tensor(index), torch.tensor(index), 0, 0

    #     elif 'test_reference' in self.mode:
    #         img_reference = Image.open(self.root + self.id_test_list[index][0]).convert('RGB')
    #         img_reference = self.transform_reference(img_reference)
    #         if self.args.crop:
    #             atten_sat = Image.open(os.path.join(self.args.resume.replace(self.args.resume.split('/')[-1],''),'attention','val',str(index)+'.png')).convert('RGB')
    #             return img_reference, torch.tensor(index), self.to_tensor(atten_sat)
    #         return img_reference, torch.tensor(index), 0

    #     elif 'test_query' in self.mode:
    #         img_query = Image.open(self.root + self.id_test_list[index][1]).convert('RGB')
    #         img_query = img_query.crop((0, img_query.size[1] // 4, img_query.size[0], img_query.size[1] // 4 * 3))
    #         img_query = self.transform_query(img_query)
    #         return img_query, torch.tensor(index), torch.tensor(index)
    #     else:
    #         print('not implemented!!')
    #         raise Exception

    # def __len__(self):
    #     if self.mode == 'train':
    #         return len(self.id_idx_list)
    #     elif 'scan_val' in self.mode:
    #         return len(self.id_test_list)
    #     elif 'test_reference' in self.mode:
    #         return len(self.id_test_list)
    #     elif 'test_query' in self.mode:
    #         return len(self.id_test_list)
    #     else:
    #         print('not implemented!')
    #         raise Exception
class Sampler_University(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source, batchsize=8,sample_num=4):
        self.data_len = len(data_source)
        self.batchsize = batchsize
        self.sample_num = sample_num

    def __iter__(self):
        list = np.arange(0,self.data_len)
        np.random.shuffle(list)
        nums = np.repeat(list,self.sample_num,axis=0)
        return iter(nums)

    def __len__(self):
        return len(self.data_source)


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    img_s,img_st,img_d,ids = zip(*batch)
    ids = torch.tensor(ids,dtype=torch.int64)
    return [torch.stack(img_s, dim=0),ids],[torch.stack(img_st,dim=0),ids], [torch.stack(img_d,dim=0),ids]

# if __name__ == '__main__':
#     transform_train_list = [
#         # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
#         transforms.Resize((256, 256), interpolation=3),
#         transforms.Pad(10, padding_mode='edge'),
#         transforms.RandomCrop((256, 256)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]


#     transform_train_list ={"satellite": transforms.Compose(transform_train_list),
#                             "train":transforms.Compose(transform_train_list)}
#     datasets = Uni_1652(root="D:/DATA_set/University-Release/train",transforms=transform_train_list,names=['satellite','drone'])
#     samper = Sampler_University(datasets,8)
#     dataloader = DataLoader(datasets,batch_size=8,num_workers=32,sampler=samper,collate_fn=train_collate_fn)
#     for data_s,data_d in dataloader:
#         print()
