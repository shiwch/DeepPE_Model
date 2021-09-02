import scheduler
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy
import time
import random
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, hidden_dim, learning_rate):
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.dim_weight = 5
        self.batch_size = 256
        self.net_type = "hac"
        self.batch_size = 64
        self.epochs = 10
        self.net_type = "hac"
        self.block_levels = 17 
        self.block = nn.ModuleList()
        
        for i in range(self.hidden_size):
            self.block.append(nn.Dropout(p = 0.0))
            self.block.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.block.append(nn.ReLU())
        
        # self.block.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        # self.block.append(nn.ReLU())

        self.inputs_forward = nn.ModuleList()
        self.inputs_backward = nn.ModuleList()
        self.blocks_forward = nn.ModuleList(nn.ModuleList())
        self.blocks_backward = nn.ModuleList(nn.ModuleList())
        self.outputs_forward = nn.ModuleList()
        self.outputs_backward = nn.ModuleList()
        for i in range(self.block_levels):
            #self.inputs_hpc.append(nn.Sequential(nn.Linear(self.input_dim + ((i > 0) + 0) * self.output_dim, self.hidden_dim), nn.ReLU()))
            #self.inputs_hac.append(nn.Sequential(nn.Linear(self.input_dim + i * self.output_dim, self.hidden_dim), nn.ReLU()))
            self.inputs_forward.append(nn.Linear(self.input_dim + i * self.output_dim, self.hidden_dim))
            self.inputs_backward.append(nn.Linear(self.input_dim + i * self.output_dim, self.hidden_dim))
            self.blocks_forward.append(self.block)
            self.blocks_backward.append(self.block)
            self.outputs_forward.append(nn.Linear(self.hidden_dim, self.output_dim))
            self.outputs_backward.append(nn.Linear(self.hidden_dim, self.output_dim))
        #self.output = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.ReLU()) 
        #self.output_forward = nn.Linear(self.hidden_dim, self.output_dim)
        #self.output_backward = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.linear1 = nn.Linear(self.dim_weight, self.dim_weight, bias=False)
        self.linear2 = nn.Linear(self.dim_weight, self.dim_weight, bias=False)
        self.linear3 = nn.Linear(self.dim_weight, self.dim_weight, bias=False)
        self.softmax = nn.Softmax(dim=1)
        #self.input_fnn = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())
        #self.output_fnn = nn.Linear(self.hidden_dim, self.output_dim * self.block_levels)

    def forward(self, input_data):
        
        prev_forward = input_data
        prev_backward = input_data
        total_out_forward = torch.Tensor().cuda()
        out_forward = torch.Tensor().cuda()
        total_out_backward = torch.Tensor().cuda()
        out_backward = torch.Tensor().cuda()
        # store the output of previous block(or blocks)
        #if self.net_type == "fnn":
        #    out = self.input_fnn(prev)

        for i in range(self.block_levels):
            
            out_forward = self.inputs_forward[i](prev_forward)
            for j in range(len(self.blocks_forward[i])):
                out_forward = self.blocks_forward[i][j](out_forward)
            out_forward = self.outputs_forward[i](out_forward)
            #out_forward = self.output_forward(out_forward)
            prev_forward = torch.cat([input_data, total_out_forward, out_forward], dim = 1)

            total_out_forward = torch.cat([total_out_forward, out_forward], dim = 1)

            out_backward = self.inputs_backward[i](prev_backward)
            for j in range(len(self.blocks_backward[i])):
                out_backward = self.blocks_backward[i][j](out_backward)
            out_backward = self.outputs_backward[i](out_backward)
            #out_backward = self.output_backward(out_backward)
            prev_backward = torch.cat([input_data, total_out_backward, out_backward], dim = 1)

            total_out_backward = torch.cat([total_out_backward, out_backward], dim = 1)
        

        total_out_forward = total_out_forward.reshape(total_out_forward.shape[0], self.output_dim, -1)
        total_out_backward = total_out_backward.reshape(total_out_backward.shape[0], self.output_dim, -1)
        total_out_backward = total_out_backward[:,:,range(16,-1,-1)]


        yt_mul_wq = self.linear1(total_out_forward.transpose(2,1))
        y_mul_wk = self.linear2(total_out_backward.transpose(2,1)).transpose(2,1)
        y_mul_wv = self.linear3(total_out_backward.transpose(2,1))

        result = torch.matmul(self.softmax(torch.matmul(yt_mul_wq, y_mul_wk)), y_mul_wv)
        result = result.reshape(total_out_backward.shape[0], -1)
        
        return result
        

class Net(nn.Module):
    def __init__(self, hidden_size, hidden_dim, learning_rate):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.batch_size = 256
        self.epochs = 60
        self.epochs_second = 40
        self.net_type = "bi-HAC-att-4-newmodel"
        self.dim_weight = 5
        self.input_data = {}
        self.output_data = {}
        self.sample_dim = 16
        self.datas = {}
        self.load_data()
        self.preprocess_data(38)
        self.split_data(38)
        self.block_sample_tag = nn.ModuleList()
        self.block_sample = nn.ModuleList()
        self.block_tag_linear = nn.ModuleList()

        # single_model = Block(self.input_dim, self.output_dim, hidden_size, hidden_dim, learning_rate)
        self.blocks_G = Block(self.input_dim, self.output_dim, hidden_size, hidden_dim, learning_rate)
        self.blocks_G_tag = Block(self.input_dim, self.output_dim, hidden_size, hidden_dim, learning_rate)
        
        self.block_tag_linear = nn.Linear(self.output_dim * self.block_levels, self.sample_dim)

        self.linear = nn.Linear(self.output_dim * self.block_levels, self.output_dim * self.block_levels)

        # self.linear = nn.Linear(self.output_dim * self.block_levels, self.sample_dim)
        # self.block.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        # self.block.append(nn.ReLU())

        

    def forward(self, input_data):
            
        out_1 = self.blocks_G(input_data)
        out_2 = self.blocks_G_tag(input_data)

        out = self.linear(out_1 + out_2)
        out_tag = self.block_tag_linear(out_2)
        out_F = (torch.norm(torch.matmul(out_2.transpose(1,0), out_1)))

        return out, out_tag, out_F
    
    def fixed(self):
        
        for i in self.blocks_G.parameters():
            i.requires_grad = False
        for i in self.linear.parameters():
            i.requires_grad = False
        for i in self.block_tag_linear.parameters():
            i.requires_grad = False 

    def load_data(self):
        for id in [2, 19, 21, 22, 31, 37, 38, 49, 50, 65, 66, 68, 72, 80, 82, 102]:
            print(str(id) +"%", end="  ")
            
            input_data = pd.read_csv('./data/Input_%03d_1984-2005.csv'%id, sep=r",", header=None, engine='python')
            output_data = pd.read_csv('./data/Output_%03d_1984-2005.csv'%id, sep=r",", header=None, engine='python')
            self.input_data[id] = input_data.iloc[:,range(16)].values.astype("float32")

            # we need 0 to 16 layers 
            self.block_levels = 17 
            self.input_dim = self.input_data[id].shape[1]
            self.output_dim = int(output_data.shape[1] / self.block_levels)
        
            total_output_dim = output_data.shape[1] 

            res = []
            for i in range(self.block_levels):
                for j in range(self.output_dim):
                    res.append(i + j * self.block_levels)  
            
            req_outputs = res
            req_outputs = [int(x) for x in req_outputs]

            self.output_data[id] = output_data.iloc[:,req_outputs].values.astype('float32')
            
        print("Load Data Finish!")
            
    def split_data(self, id):
        train_idx = range(0, 58365-1)
        val_idx = range(58365, 61290-1)
        test_idx = range(61290, 64207)
        train_input_data = torch.Tensor(self.input_data[id])
        train_output_data = torch.Tensor(self.output_data[id])

        self.train_x, self.train_y = train_input_data[train_idx,:], train_output_data[train_idx,:]
        self.val_x, self.val_y = train_input_data[val_idx,:], train_output_data[val_idx,:]
        self.test_x, self.test_y = train_input_data[test_idx,:], train_output_data[test_idx,:]
        self.train_xx = {}
        self.train_yy = {}
        self.val_xx = {}
        self.val_yy = {}
        self.test_xx = {}
        self.test_yy = {}
        self.val_all = torch.Tensor()
        self.test_all = torch.Tensor()
        self.train_all = torch.Tensor()
        for i in [2, 19, 21, 22, 31, 37, 38, 49, 50, 65, 66, 68, 72, 80, 82, 102]:
            self.train_xx[i] = torch.Tensor(self.input_data[i][train_idx,:])
            self.train_yy[i] = torch.Tensor(self.output_data[i][train_idx,:])
            self.val_xx[i] = torch.Tensor(self.input_data[i][val_idx,:])
            self.val_yy[i] = torch.Tensor(self.output_data[i][val_idx,:])
            self.test_xx[i] = torch.Tensor(self.input_data[i][test_idx,:])
            self.test_yy[i] = torch.Tensor(self.output_data[i][test_idx,:])
            self.val_all = torch.cat([self.val_all, self.val_xx[i]],dim=1)
            self.test_all = torch.cat([self.test_all, self.test_xx[i]], dim=1)
            self.train_all = torch.cat([self.train_all, self.train_xx[i]], dim=1)

        print("Split Data Finish!")


    def preprocess_data(self, id):
        # normalize features
        self.preprocessor = Pipeline([('stdscaler', StandardScaler()), ('minmax', MinMaxScaler(feature_range=(0, 1)))])
        self.preprocessor.fit(numpy.concatenate((self.input_data[id], self.output_data[id]), axis=1))
        for i in [2, 19, 21, 22, 31, 37, 38, 49, 50, 65, 66, 68, 72, 80, 82, 102]:

            temp = self.preprocessor.transform(numpy.concatenate((self.input_data[i], self.output_data[i]), axis=1))
            self.input_data[i] = temp[:,range(self.input_dim)]
            self.output_data[i] = temp[:,range(self.input_dim, temp.shape[1])]
        '''
        self.preprocessor = Pipeline([('stdscaler', StandardScaler()), ('minmax', MinMaxScaler(feature_range=(0, 1)))])
        
        self.preprocessor.fit(self.output_data[id])
        for i in [2, 19, 21, 22, 31, 37, 38, 49, 50, 65, 66, 68, 72, 80, 82, 102]:
            # print(i, self.output_data[i].shape)
            self.output_data[i] = self.preprocessor.transform(self.output_data[i])
        '''
    def change(self, train_data_id):
        self.train_x = self.train_xx[train_data_id]
        self.train_y = self.train_yy[train_data_id]
        self.val_x = self.val_xx[train_data_id]
        self.val_y = self.val_yy[train_data_id]
        self.test_x = self.test_xx[train_data_id]
        self.test_y = self.test_yy[train_data_id]
    
    def merge_data(self):
        val_id = {2:0, 19:1, 21:2, 22:3, 31:4, 37:5, 38:6, 49:7, 50:8, 65:9, 66:10, 68:11, 72:12, 80:13, 82:14, 102:15}
        train_data = torch.Tensor() #= torch.cat([self.train_x, self.train_y], dim = 1)
        #train_data = torch.cat([train_data, 6 * torch.ones(train_data.shape[0]).reshape(train_data.shape[0],-1)], dim = 1)
        for i in [2, 19, 21, 22, 31, 37, 38, 49, 50, 65, 66, 68, 72, 80, 82, 102]:
            temp = torch.cat([self.train_xx[i], self.train_yy[i]], dim = 1)
            temp = torch.cat([temp, val_id[i] * torch.ones(temp.shape[0]).reshape(temp.shape[0],-1)], dim = 1)
            train_data = torch.cat([train_data, temp], dim = 0)
        index = [x for x in range(train_data.shape[0])]
        print(len(index))
        #print(train_data.shape)

        random.shuffle(index)
        train_data = train_data[index,:]
        print(self.train_x.shape)
        self.train_x = train_data[:,range(self.train_x.shape[1])]
        self.train_y = train_data[:,range(self.train_x.shape[1], train_data.shape[1]-1)]
        self.tag = train_data[:, train_data.shape[1] - 1].long()
        self.tag = self.tag.reshape(self.tag.shape[0])
        print(self.train_x.shape)
        print(self.train_y.shape)

        print(index[0:10])



def write_file(file_name, string):
    f = open(file_name, 'a+')
    f.write(string)        

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

data_range = [2925,5842,8759,11676,14601,17518,20435,23352,26273,29190,32107,35024,37941,40858,43775,46692,49617,52531,55448,58365,61290,64207]
val_data_id_start = [0, 5842, 11676, 17518, 23352, 29190, 35024, 40858, 46692, 52531, 58365]
val_middle_test = [2925, 8759, 14601, 20435, 26273, 32107, 37941, 43775, 49617, 55448, 61290]
test_data_id_end = [5842, 11676, 17518, 23352, 29190, 35024, 40858, 46692, 52531, 58365, 64207]

train_time = 0.0


model = Net(2, 4, 0.001).cuda()
model.merge_data()
model.loss_func = nn.MSELoss()
model.loss_func_tag = nn.CrossEntropyLoss()

#model.loss_func = nn.MSELoss()
model.optimizer = torch.optim.Adam(model.parameters() ,lr=0.001)
#model.optimizer = torch.optim.SGD(model.parameters(), 0.0001)
after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, float(model.epochs), eta_min=0.0001)
#after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
scheduler_ = scheduler.GradualWarmupScheduler(model.optimizer, multiplier=1, total_epoch=1, after_scheduler=after_scheduler)
    
print(get_parameter_number(model))

'''
Vals_MAE_List = {}
Vals_R2_List = {}
Vals_Explain_List = {}
Vals_Loss_List = {}
Vals_Loss_Rmse_List = {}
'''
Vals_MAE_List = []
Vals_R2_List = []
Vals_Explain_List = []
Vals_Loss_List = []
Vals_Loss_Rmse_List = []
'''
for i in [2, 19, 21, 22, 31, 37, 38, 49, 50, 65, 66, 68, 72, 80, 82, 102]:
    Vals_MAE_List[i] = []
    Vals_R2_List[i] = []
    Vals_Explain_List[i] = []
    Vals_Loss_List[i] = []
    Vals_Loss_Rmse_List[i] = []
'''
Train_Loss_Total_List = []
Train_Loss_List = []
Train_Loss_F_List = []
Train_Loss_Tag_List = []
Train_Loss_Rmse_List = []
Train_learining_rate = []


start = time.time()

for epoch in range(model.epochs):
    #print(epoch)
    scheduler_.step()
    Train_learining_rate.append(model.optimizer.state_dict()['param_groups'][0]['lr'])
    #loss_value = 0
    #Val_Loss_List = []
    '''
    Val_predict = model(model.val_x.cuda()).cpu()
    Val_loss = model.loss_func(Val_predict.cuda(), model.val_y.cuda())
    for j in range(85):            
        Vals_R2_List.append(r2_score(Val_predict[:,j].tolist(), model.val_y[:,j].tolist()))
        Vals_MAE_List.append(mean_absolute_error(Val_predict[:,j].tolist() , model.val_y[:,j].tolist()))
    Vals_Loss_List.append(Val_loss.cpu().detach().numpy())
    Vals_Loss_Rmse_List.append(torch.sqrt(Val_loss).cpu().detach().numpy())
    '''
    loss_basic, loss_value, loss_tag_value, loss_f, loss_rmse = 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(model.train_x.shape[0] // model.batch_size):
        model.optimizer.zero_grad()
        idx = range(i*model.batch_size, (i+1)*model.batch_size)
        if((i+1)*model.batch_size >= model.train_x.shape[0]):
            idx = range(i*model.batch_size, model.train_x.shape[0])
        
        model_out, model_tag, model_f = model(model.train_x[idx,:].cuda())
        
        loss = model.loss_func(model_out, model.train_y[idx,:].cuda())
        loss_tag = model.loss_func_tag(model_tag, model.tag[idx].cuda())
        
        loss_basic = loss_basic + loss.cpu().detach().numpy()
        loss_f = loss_f + model_f.cpu().detach().numpy()
        loss_tag_value = loss_tag_value + loss_tag.cpu().detach().numpy()

        loss = loss + 0.1 * model_f + 0.005 * loss_tag
        loss.backward()
        model.optimizer.step()
        loss_value = loss_value + loss.cpu().detach().numpy()

        loss_rmse = loss_rmse + torch.sqrt(loss).cpu().detach().numpy()
        
    Train_Loss_List.append(loss_basic / (model.train_x.shape[0] // model.batch_size))
    Train_Loss_Total_List.append(loss_value / (model.train_x.shape[0] // model.batch_size))
    Train_Loss_F_List.append(loss_f / (model.train_x.shape[0] // model.batch_size))
    Train_Loss_Tag_List.append(loss_tag_value / (model.train_x.shape[0] // model.batch_size))
    Train_Loss_Rmse_List.append(loss_rmse / (model.train_x.shape[0] // model.batch_size))

    #scheduler_.step()

    #scheduler_.step(Val_loss)
    
end = time.time()
print("first_train_time: " + str(end - start))

train_time = train_time + end - start

torch.save(model.state_dict(), model.net_type+"_params_list.pth")

write_file("./migration_training/"+model.net_type+"/"+"first_train_loss_total.txt", ",".join([str(x) for x in Train_Loss_Total_List]))
write_file("./migration_training/"+model.net_type+"/"+"first_train_loss.txt", ",".join([str(x) for x in Train_Loss_List]))
write_file("./migration_training/"+model.net_type+"/"+"first_train_loss_f.txt", ",".join([str(x) for x in Train_Loss_F_List]))
write_file("./migration_training/"+model.net_type+"/"+"first_train_loss_tag.txt", ",".join([str(x) for x in Train_Loss_Tag_List]))
write_file("./migration_training/"+model.net_type+"/"+"first_train_loss_rmse.txt", ",".join([str(x) for x in Train_Loss_Rmse_List]))
write_file("./migration_training/"+model.net_type+"/"+"first_train_lr.txt", ",".join([str(x) for x in Train_learining_rate]))


val_id = {2:0, 19:1, 21:2, 22:3, 31:4, 37:5, 38:6, 49:7, 50:8, 65:9, 66:10, 68:11, 72:12, 80:13, 82:14, 102:15}

for id in [2, 19, 21, 22, 31, 37, 38, 49, 50, 65, 66, 68, 72, 80, 82, 102]:

    test_MAE_List = []
    test_R2_List = []
    test_Explain_List = []
    test_Loss_List = []
    test_Loss_F_List = []
    test_Loss_Total_List = []
    test_Loss_Tag_List = []
    test_Loss_Rmse_List = []
    Train_Loss_List = []
    Train_Loss_F_List = []
    Train_Loss_Tag_List = []
    Train_Loss_Total_List = []
    Train_Loss_Rmse_List = []
    Train_learining_rate = []

    # model = Net(2, 8, 0.001).cuda()
    if os.path.exists(model.net_type+"_params_list.pth"):
        model.load_state_dict(torch.load(model.net_type+"_params_list.pth"))
        print("load params succeed!")
    model.change(id)
    model.fixed()
    model.optimizer = torch.optim.Adam(model.parameters() ,lr=0.0001)
    after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, float(10), eta_min=0.00001)
    #after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, mode='min', factor=0.8, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    scheduler_ = scheduler.GradualWarmupScheduler(model.optimizer, multiplier=1, total_epoch=1, after_scheduler=after_scheduler)

    for epoch in range(model.epochs_second):
        scheduler_.step()
        Train_learining_rate.append(model.optimizer.state_dict()['param_groups'][0]['lr'])
        #loss_value = 0
        #Val_Loss_List = []
        '''
        Val_predict = model(model.val_x.cuda()).cpu()
        Val_loss = model.loss_func(Val_predict.cuda(), model.val_y.cuda())
        for j in range(85):            
            Vals_R2_List.append(r2_score(Val_predict[:,j].tolist(), model.val_y[:,j].tolist()))
            Vals_MAE_List.append(mean_absolute_error(Val_predict[:,j].tolist() , model.val_y[:,j].tolist()))
        Vals_Loss_List.append(Val_loss.cpu().detach().numpy())
        Vals_Loss_Rmse_List.append(torch.sqrt(Val_loss).cpu().detach().numpy())
        '''
        
        loss_value, loss_f, loss_tag_value, loss_basic = 0.0, 0.0, 0.0, 0.0
        
        start = time.time()
        for i in range(model.train_x.shape[0] // model.batch_size):
            model.optimizer.zero_grad()
            idx = range(i*model.batch_size, (i+1)*model.batch_size)
            if((i+1)*model.batch_size >= model.train_x.shape[0]):
                idx = range(i*model.batch_size, model.train_x.shape[0])
            
            model_out, model_tag, model_f = model(model.train_x[idx,:].cuda())
        
            loss = model.loss_func(model_out, model.train_y[idx,:].cuda())
            loss_tag = model.loss_func_tag(model_tag, val_id[id] * torch.ones(model_tag.shape[0]).long().cuda())

            loss_basic = loss_basic + loss.cpu().detach().numpy()
            loss_f = loss_f + model_f.cpu().detach().numpy()
            loss_tag_value = loss_tag_value + loss_tag.cpu().detach().numpy()

            loss = loss + 0.1 * model_f + 0.005 * loss_tag
            loss.backward()
            model.optimizer.step()
            loss_value = loss_value + loss.cpu().detach().numpy()
            loss_rmse = loss_rmse + torch.sqrt(loss).cpu().detach().numpy()

        Train_Loss_List.append(loss_basic / (model.train_x.shape[0] // model.batch_size))
        Train_Loss_F_List.append(loss_f / (model.train_x.shape[0] // model.batch_size))
        Train_Loss_Tag_List.append(loss_tag_value / (model.train_x.shape[0] // model.batch_size))
        Train_Loss_Total_List.append(loss_value / (model.train_x.shape[0] // model.batch_size))
        Train_Loss_Rmse_List.append(loss_rmse / (model.train_x.shape[0] // model.batch_size))

        end = time.time()

        train_time = train_time + end - start



        start = time.time()
        
        y_hat, y_tag, y_f = model(model.test_x.cuda())
        
        end = time.time()
        print("second_train_time "+str(epoch)+"th : " + str(end - start))
        
        test_loss = model.loss_func(y_hat, model.test_y.cuda())
        test_loss_tag = model.loss_func_tag(y_tag, val_id[id] * torch.ones(y_tag.shape[0]).long().cuda())
        
        test_Loss_F_List.append(y_f.cpu().detach().numpy())
        test_Loss_Tag_List.append(test_loss_tag.cpu().detach().numpy())
        test_Loss_Total_List.append((test_loss + 0.1 * y_f + 0.15 * test_loss_tag).cpu().detach().numpy())
        test_Loss_Rmse_List.append(torch.sqrt(test_loss + 0.1 * y_f + 0.15 * test_loss_tag).cpu().detach().numpy())
        test_Loss_List.append(test_loss.cpu().detach().numpy())

        for j in range(85):            
            test_R2_List.append(r2_score(y_hat[:,j].tolist(), model.test_y[:,j].tolist()))
            test_MAE_List.append(mean_absolute_error(y_hat[:,j].tolist(), model.test_y[:,j].tolist()))
        test_Loss_List.append(test_loss.cpu().detach().numpy())
        test_Loss_Rmse_List.append(torch.sqrt(test_loss).cpu().detach().numpy())
        
        #print(y_hat.detach())
        #print("***************************")
        #print(y_hat)
        if epoch == model.epochs_second-1:
            test_pred = numpy.concatenate((model.test_x.numpy(), y_hat.cpu().detach().numpy()), axis = 1)
            test_orig = numpy.concatenate((model.test_x.numpy(), model.test_y.numpy()), axis = 1)
            test_pred_trans = model.preprocessor.inverse_transform(test_pred)
            test_orig_trans = model.preprocessor.inverse_transform(test_orig)
            pred_df = pd.DataFrame(test_pred_trans)
            orig_df = pd.DataFrame(test_orig_trans)
            pred_df.to_csv("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_pred_"+"%03d"%id+".csv", index=False)
            orig_df.to_csv("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_orig_"+"%03d"%id+".csv", index=False)
        
    

    for j in range(model.epochs_second):
        # string = ",".join([str(x) for x in Vals_R2_List[i][j*85:(j+1)*85]])
        string = ",".join([str(x) for x in test_R2_List[j*85:(j+1)*85]])
        write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_test_r2_list.txt", string +'\n')
        
        string = ",".join([str(x) for x in test_MAE_List[j*85:(j+1)*85]])
        write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_test_mae_list.txt", string +'\n')
    
    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_test_r2_list"+".txt", '\n')
    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_test_mae_list.txt",'\n')   
    
    print(len([x for x in test_R2_List[j*85:(j+1)*85] if x < 0]), end = " ")
    
    
    #write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_test_loss.txt",'\n')
    #write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_test_loss_rmse.txt",'\n')
    

    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_test_loss_f.txt", ",".join([str(x) for x in test_Loss_F_List]) +'\n')
    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_test_loss_tag.txt", ",".join([str(x) for x in test_Loss_Tag_List]) +'\n')
    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_test_loss.txt", ",".join([str(x) for x in test_Loss_List]) +'\n')
    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_test_loss_rmse.txt", ",".join([str(x) for x in test_Loss_Rmse_List]) +'\n')
    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_test_loss_total.txt", ",".join([str(x) for x in test_Loss_Total_List]) +'\n')

    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_train_loss_f.txt", ",".join([str(x) for x in Train_Loss_F_List]))
    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_train_loss_tag.txt", ",".join([str(x) for x in Train_Loss_Tag_List]))
    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_train_loss.txt", ",".join([str(x) for x in Train_Loss_List]))
    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_train_loss_total.txt", ",".join([str(x) for x in Train_Loss_Total_List]))
    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_train_loss_rmse.txt", ",".join([str(x) for x in Train_Loss_Rmse_List]))

    write_file("./migration_training/"+model.net_type+"/"+str(id)+"/"+"_train_lr.txt", ",".join([str(x) for x in Train_learining_rate]))
    # torch.save(model.state_dict(), model.net_type+"_params_list.pth")
    

print("total_train_time: "+str(train_time))