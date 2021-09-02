import scheduler
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy
import math
import time
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error


class Net(nn.Module):
    def __init__(self, hidden_size, hidden_dim, learning_rate):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_size
        self.batch_size = 64
        self.epochs = 100
        self.net_type = "bi-HAC-att-4"
        self.dim_weight = 5
        self.input_data = {}
        self.output_data = {}
        self.datas = {}
        self.load_data()
        self.preprocess_data(38)
        self.split_data(38)
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
        self.linear4 = nn.Linear(self.output_dim * self.block_levels, self.output_dim * self.block_levels)
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
        result = self.linear4(result.reshape(total_out_backward.shape[0], -1))

        return result
        

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
        for i in [2, 19, 21, 22, 31, 37, 38, 49, 50, 65, 66, 68, 72, 80, 82, 102]:
            self.train_xx[i] = torch.Tensor(self.input_data[i][train_idx,:])
            self.train_yy[i] = torch.Tensor(self.output_data[i][train_idx,:])
            self.val_xx[i] = torch.Tensor(self.input_data[i][val_idx,:])
            self.val_yy[i] = torch.Tensor(self.output_data[i][val_idx,:])
            self.test_xx[i] = torch.Tensor(self.input_data[i][test_idx,:])
            self.test_yy[i] = torch.Tensor(self.output_data[i][test_idx,:])
        
        print("Split Data Finish!")

    def split_data_again(self, id, val_range_start_id, val_middle_test, test_range_end_id):
        train_idx1 = range(0, val_range_start_id)
        train_idx2 = range(test_range_end_id, 64207)

        val_idx = range(val_range_start_id, val_middle_test)
        test_idx = range(val_middle_test, test_range_end_id)
        train_input_data = torch.Tensor(self.input_data[id])
        train_output_data = torch.Tensor(self.output_data[id])
        
        self.train_x, self.train_y = torch.cat([train_input_data[train_idx1,:], train_input_data[train_idx2,:]], dim = 0), torch.cat([train_output_data[train_idx1,:], train_output_data[train_idx2,:]], dim = 0)
        #print(self.train_x.shape)
        self.val_x, self.val_y = train_input_data[val_idx,:], train_output_data[val_idx,:]
        self.test_x, self.test_y = train_input_data[test_idx,:], train_output_data[test_idx,:]
        
        '''
        self.train_xx = {}
        self.train_yy = {}
        self.val_xx = {}
        self.val_yy = {}
        self.test_xx = {}
        self.test_yy = {}
        
        for i in [2, 19, 21, 22, 31, 37, 38, 49, 50, 65, 66, 68, 72, 80, 82, 102]:
            
            self.train_xx[i] = torch.cat([self.input_data[i][train_idx1,:], self.input_data[i][train_idx2,:]], dim = 0)
            self.train_yy[i] = torch.cat([self.output_data[i][train_idx1,:], self.output_data[i][train_idx2,:]], dim = 0)
            self.val_xx[i] = torch.Tensor(self.input_data[i][val_idx,:])
            self.val_yy[i] = torch.Tensor(self.output_data[i][val_idx,:])
            self.test_xx[i] = torch.Tensor(self.input_data[i][test_idx,:])
            self.test_yy[i] = torch.Tensor(self.output_data[i][test_idx,:])
        '''
        print("Split Data Again Finish!")

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

for test_num in range(0,11):

    model = Net(2, 4, 0.001).cuda()
    model.split_data_again(38, val_data_id_start[test_num], val_middle_test[test_num], test_data_id_end[test_num])
    model.loss_func = nn.MSELoss()
    #model.loss_func = nn.MSELoss()
    model.optimizer = torch.optim.Adam(model.parameters() ,lr=0.001)
    #model.optimizer = torch.optim.SGD(model.parameters(), 0.0001)
    after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, float(model.epochs), eta_min=0.00001)
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
    Train_Loss_List = []
    Train_learining_rate = []


    start = time.time()
    
    for epoch in range(model.epochs):
        #print(epoch)
        scheduler_.step()
        Train_learining_rate.append(model.optimizer.state_dict()['param_groups'][0]['lr'])
        #loss_value = 0
        #Val_Loss_List = []
        
        Val_predict = model(model.val_x.cuda()).cpu()
        Val_loss = model.loss_func(Val_predict.cuda(), model.val_y.cuda())
        for j in range(85):            
            Vals_R2_List.append(r2_score(Val_predict[:,j].tolist(), model.val_y[:,j].tolist()))
            Vals_MAE_List.append(mean_absolute_error(Val_predict[:,j].tolist() , model.val_y[:,j].tolist()))
        Vals_Loss_List.append(Val_loss.cpu().detach().numpy())
        Vals_Loss_Rmse_List.append(torch.sqrt(Val_loss).cpu().detach().numpy())
        
        
        loss_value = 0
        
        for i in range(model.train_x.shape[0] // model.batch_size):
            model.optimizer.zero_grad()
            idx = range(i*model.batch_size, (i+1)*model.batch_size)
            if((i+1)*model.batch_size >= model.train_x.shape[0]):
                idx = range(i*model.batch_size, model.train_x.shape[0])
            
            model_out = model(model.train_x[idx,:].cuda())
            loss = model.loss_func(model_out, model.train_y[idx,:].cuda())
            loss_value = loss_value + loss.cpu().detach().numpy()

            loss.backward()
            
            model.optimizer.step()
        
        Train_Loss_List.append(loss_value / (model.train_x.shape[0] // model.batch_size))
        #scheduler_.step()

        #scheduler_.step(Val_loss)
        
    end = time.time()
    print("train_time "+str(test_num)+"th : " + str(end - start))
    train_time = train_time + end - start
    
    
    start = time.time()
    y_hat = model(model.test_x.cuda()).cpu()
    end = time.time()
    print("test_train_time "+str(test_num)+"th : " + str(end - start))
    test_pred = numpy.concatenate((model.test_x.numpy(), y_hat.detach().numpy()), axis = 1)
    test_orig = numpy.concatenate((model.test_x.numpy(), model.test_y.numpy()), axis = 1)
    #print(y_hat.detach())
    #print("***************************")
    #print(y_hat)
    test_pred_trans = model.preprocessor.inverse_transform(test_pred)
    test_orig_trans = model.preprocessor.inverse_transform(test_orig)
    pred_df = pd.DataFrame(test_pred_trans)
    orig_df = pd.DataFrame(test_orig_trans)
    pred_df.to_csv("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_pred_"+"038"+".csv", index=False)
    orig_df.to_csv("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_orig_"+"038"+".csv", index=False)

    # torch.save(model.state_dict(), model.net_type+"_params_list.pth")

        

    # for i in [2, 19, 21, 22, 31, 37, 38, 49, 50, 65, 66, 68, 72, 80, 82, 102]:
    string = ",".join([str(x) for x in Vals_Loss_List])
    write_file("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_val_loss.txt", string +'\n')

    string = ",".join([str(x) for x in Vals_Loss_Rmse_List])
    write_file("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_val_loss_rmse.txt", string +'\n')

    for j in range(model.epochs):
        # string = ",".join([str(x) for x in Vals_R2_List[i][j*85:(j+1)*85]])
        string = ",".join([str(x) for x in Vals_R2_List[j*85:(j+1)*85]])
        write_file("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_val_r2_list.txt", string +'\n')
        
        string = ",".join([str(x) for x in Vals_MAE_List[j*85:(j+1)*85]])
        write_file("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_val_mae_list.txt", string +'\n')
        
    print(len([x for x in Vals_R2_List[j*85:(j+1)*85] if x < 0]), end = " ")
    write_file("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_val_r2_list"+".txt", '\n')
    write_file("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_val_mae_list.txt",'\n')
    write_file("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_val_loss.txt",'\n')
    write_file("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_val_loss_rmse.txt",'\n')

        
    write_file("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_train_loss.txt", ",".join([str(x) for x in Train_Loss_List]))
    write_file("./cross_training/"+model.net_type+"/"+str(test_num)+"/"+"_train_lr.txt", ",".join([str(x) for x in Train_learining_rate]))

print("total_train_time: "+str(train_time))