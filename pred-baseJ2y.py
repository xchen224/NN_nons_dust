import numpy as np
import pickle, sys
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


allvars   = ['ext','abs','F11','F22','F33','F44','F12','F34']
allvarsname   = ['F_{33}/F_{11}','F_{44}/F_{11}','F_{12}/F_{11}','F_{34}/F_{11}']

# F33 & F44
'''
task = 0
mlp_layers = 4
mlp_dim = 768 #768
max_epochs = 1000
lr = 1e-4
min_lr = 1e-5
'''
task = 4
mlp_layers = 9 #4
mlp_dim = 641 #768
max_epochs = 3000


useJk  = False


data = np.load('./alldata/split_data_rat'+allvars[task]+'.npz')
train_x0 = data['train_x']
test_x0  = data['test_x']
train_y1 = data['train_y']
test_y1  = data['test_y']
J_mean1  = data['J_mean']
J_std1   = data['J_std']
data = np.load('./alldata/split_data_rat'+allvars[task+1]+'.npz')
train_y2 = data['train_y']
test_y2  = data['test_y']
J_mean2  = data['J_mean']
J_std2   = data['J_std']

print('before:', train_y1.shape, np.max(train_y1,axis=0), np.min(train_y1,axis=0), J_mean1, J_std1)

if useJk:
    train_y1[:,1] = (train_y1[:,1] * J_std1[0][0] + J_mean1[0][0]) / np.exp(train_x0[:,2])
    train_y2[:,1] = (train_y2[:,1] * J_std1[0][0] + J_mean1[0][0]) / np.exp(train_x0[:,2])
    test_y1[:,1] = (test_y1[:,1] * J_std1[0][0] + J_mean1[0][0]) / np.exp(test_x0[:,2])
    test_y2[:,1] = (test_y2[:,1] * J_std1[0][0] + J_mean1[0][0]) / np.exp(test_x0[:,2])
    J_mean1[0][0] = np.mean(train_y1[:,1])
    J_mean2[0][0] = np.mean(train_y2[:,1])
    J_std1[0][0] = np.std(train_y1[:,1])
    J_std2[0][0] = np.std(train_y2[:,1])
    train_y1[:,1] = (train_y1[:,1] -J_mean1[0][0]) / J_std1[0][0]
    train_y2[:,1] = (train_y2[:,1] -J_mean2[0][0]) / J_std2[0][0]
    test_y1[:,1] = (test_y1[:,1] -J_mean1[0][0]) / J_std1[0][0]
    test_y2[:,1] = (test_y2[:,1] -J_mean2[0][0]) / J_std2[0][0]
    
    print('after useJk:', np.max(train_y1,axis=0), np.min(train_y1,axis=0), J_mean1, J_std1)
    
Usey    = np.hstack((np.vstack((train_y1,test_y1)), np.vstack((train_y2,test_y2))))
UseX    = np.vstack((train_x0,test_x0))
J_mean  = np.hstack((J_mean1, J_mean2))
J_std   = np.hstack((J_std1, J_std2))
num_examples, num_features = UseX.shape
num_tasks = 2

print('final:', UseX.shape, Usey.shape, Usey[Usey[:,-1]>0,:], np.max(Usey, axis=0))


#dir_name  = 'ratF12F34_rawloss_addrat_nosfvalid_rmn_ybaseJac'
dir_name  = '2ratF33F44_rmn_ybaseJac'


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
lr = 0.19945623777164 #0.01 #0.000269 #0.01
min_lr = 0.0004507414

x_mean = train_x0.mean(axis=0, keepdims=True)
x_std  = train_x0.std(axis=0, keepdims=True)

# Split dataset
train_x, test_x, train_y, test_y = train_test_split(UseX, Usey, test_size=0.2)
    
train_x = (train_x - x_mean) / x_std
test_x = (test_x - x_mean) / x_std
    
train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)
test_x = torch.Tensor(test_x)
test_y = torch.Tensor(test_y)
    

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
    
    # Transform data
    x_mean = torch.Tensor(x_mean).cuda(device)
    x_std = torch.Tensor(x_std).cuda(device)
    J_mean = torch.Tensor(J_mean).cuda(device)
    J_std = torch.Tensor(J_std).cuda(device)
    
else:
    x_mean = torch.Tensor(x_mean)
    x_std = torch.Tensor(x_std)
    J_mean = torch.Tensor(J_mean)
    J_std = torch.Tensor(J_std)


def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

# Construct Pytorch/Skorch model
class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix
    
    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
  
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
    
        input_dim = num_features
        for i in range(mlp_layers-1):
                self.add_module('fc_' + str(i), nn.Linear(input_dim, mlp_dim))
                input_dim = mlp_dim
        self.fc = AttrProxy(self, 'fc_')
        self.fc1 = nn.Linear(input_dim, num_tasks)
    
    def forward(self, x):
        for i in range(mlp_layers-1):
                fc = self.fc.__getitem__(i)
                x = torch.tanh(fc(x))
        x = self.fc1(x)
        return x


from skorch.callbacks import Checkpoint, LoadInitState, LRScheduler
cp1 = Checkpoint(dirname='./'+dir_name+'/' ) #ratF12F34_rawloss_addrat_rmn_ybaseJac

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net_regr = NeuralNetRegressor(
    Net,
    max_epochs=max_epochs,
    batch_size=1024,
    lr=lr,
    device=device,
)

net_regr.initialize()
net_regr.load_params(checkpoint=cp1)
#print(net_regr.module_.state_dict())

train_score = net_regr.score(train_x.cpu(), train_y[:,[0,4]].cpu())
test_score = net_regr.score(test_x.cpu(), test_y[:,[0,4]].cpu())
print(train_score, test_score)

# read Tmatrix data
tmaty = {}
tmatJ2 = {}
with open('../Tmat_Jac/tmat_Jacall_norm_intg.pickle', 'rb') as handle:
    data2 = pickle.load(handle)
for key in data2['value'].keys():
    tmaty[key] = data2['value'][key]
    tmatJ2[key] = np.reshape( np.append( data2['Jacmi'][key], data2['Jacmr'][key] ), (2,len(data2['value'][key])) )
cho_idx = np.array(data2['idx'])
print('cho_idx',len(cho_idx))

def get_ut_jacobian(net, x):
    net.to(device)
    batch_size = x.shape[0]
    xt = x 
    xt.requires_grad_(True)
    #y = torch.exp(net(xt))
    y = net(xt)
    y_pred1 = torch.reshape(y[:,0], (batch_size, 1))
    y_pred2 = torch.reshape(y[:,1], (batch_size, 1))
    #print(y.detach().cpu().numpy().shape)
    Jx1 = torch.autograd.grad(outputs=y_pred1, inputs=xt, grad_outputs=torch.eye(batch_size,).to(xt.device),
                       create_graph=True, retain_graph=True, only_inputs=True)[0]
    Jx2 = torch.autograd.grad(outputs=y_pred2, inputs=xt, grad_outputs=torch.eye(batch_size,).to(xt.device),
                       create_graph=True, retain_graph=True, only_inputs=True)[0]
    return y.detach().cpu().numpy(), Jx1.detach().cpu().numpy(), Jx2.detach().cpu().numpy(), xt.detach().cpu().numpy()





# datapoints not training
# T-matrix data
filename = '../Tmat_Jac/tmat_fine_n_intg.pickle'
with open(filename, 'rb') as f:
    TMdata = pickle.load(f)
X_ori = np.reshape(np.array(TMdata['x']),(int(len(TMdata['x'])/4),4))
idx   = np.where( (np.abs(X_ori[:,-1]-1.291429) > 1.0e-6 ) & (np.abs(X_ori[:,-1]-1.33) > 1.0e-6 ) )[0]
Xfine    = X_ori[idx,:]
Xfine[:,1] = np.log(Xfine[:,1])
Xfine[:,2] = np.log(Xfine[:,2])
xk    = np.exp(Xfine[:,2])
xn    = Xfine[:,3]
yfine = dict()
Jfine = dict()
for var in TMdata['value'].keys():
   y_ori = TMdata['value'][var]
   J_ori = np.reshape( np.append( TMdata['Jacmi'][var], TMdata['Jacmr'][var] ), (2,len(TMdata['Jacmr'][var])) )
   yfine[var] = y_ori[idx]
   Jfine[var] = J_ori[:,idx].T

print(Xfine)
Xp = torch.Tensor(Xfine).cuda(device)
Xp = (Xp-x_mean)/x_std
print('Xp',Xp)
with torch.no_grad():
    pred_y = (net_regr.module_(Xp)).cpu().numpy()
print('pred_y',pred_y)

from scatter_plot import scatter
for i in range(2):
   key = vars[i]
   fig = plt.figure(figsize=(4.5,3.5))
   ax = fig.add_subplot(111)
   ymi = np.nanmin(np.append(yfine[key]/yfine['F11'],pred_y[:,i]))
   yma = np.nanmax(np.append(yfine[key]/yfine['F11'],pred_y[:,i]))
   paths, slope, intercept = scatter(ax, yfine[key]/yfine['F11'], pred_y[:,i], thresh=1.e-4,color='b',label_p = 'lower right',fsize=12)
   plt.xlim(ymi, yma)
   plt.ylim(ymi, yma)
   #ax.set_xlabel(r'T-matrix c$_{ext}$ ($\mu$m$^{-1}$)', fontsize=12)
   #ax.set_ylabel(r'NN Predict c$_{ext}$ ($\mu$m$^{-1}$)', fontsize=12)
   ax.set_xlabel(r'Tmatrix '+key+'/F11' )
   ax.set_ylabel(r'NN Predict '+vars[i]+'/F11' )
   #xticks_arr = np.linspace(ymi, yma, 6)
   #ax.set_xticks(xticks_arr)
   #ax.set_xticklabels( ['{:.2f}'.format(i*1e3) for i in xticks_arr] )
   #ax.set_yticks(xticks_arr)
   #ax.set_yticklabels( ['{:.2f}'.format(i*1e3) for i in xticks_arr] )
   #plt.text(ymi,yma+0.01*(yma-ymi), 'e-3')
   #plt.text(yma+0.01*(yma-ymi),ymi, 'e-3')
   plt.tight_layout()
   plt.savefig("img/TMlogX_scatter_2rat"+key+"_2y_"+save_name+"_data2.png", dpi=300, facecolor='w', edgecolor='w')
   #plt.savefig("img/TMlogX_scatter_cext_rmn_ybaseJac_data2.png", dpi=300, facecolor='w', edgecolor='w')
   plt.close()

from torch.utils.data import DataLoader, TensorDataset
pred_X_t = torch.tensor(Xfine).float()
loader = DataLoader( pred_X_t, batch_size=5000)
all_yp = np.array([])
all_Jp1 = np.array([])
all_Jp2 = np.array([])
for data in loader:
    Xp = data.cuda(device)
    Xp = (Xp-x_mean)/x_std
    pred_y, pred_J1, pred_J2, xt = get_ut_jacobian(net_regr.module_, Xp)
    print(pred_y.shape, pred_J1.shape)
    if len(all_yp) == 0:
        all_yp = pred_y
        all_Jp1 = pred_J1
        all_Jp2 = pred_J2
    else:
        all_yp = np.vstack((all_yp, pred_y))
        all_Jp1 = np.vstack((all_Jp1, pred_J1))
        all_Jp2 = np.vstack((all_Jp2, pred_J2))
    print(all_yp.shape, all_Jp1.shape)
all_Jp = np.hstack((all_Jp1, all_Jp2))

from scatter_plot import scatter

xx = ['lnk','n']
xlab = ['lnk','n']
xx_use = {}
cb_t   = {}
xx_use['lnk'] = Xfine[:,2]
xx_use['n']   = Xfine[:,3]
cb_t['lnk']   = xx_use['lnk'][:181*14*5:181*14]
cb_t['n']     = xx_use['n'][::181*14*5]
rat = True
for i in range(2):
    for j in range(2):
        fig = plt.figure(figsize=(5,3.5))
        ax = fig.add_subplot(111)
        if rat:
            #tmatJrat = 0.0
            tmatJrat = 1.0 / (yfine['F11'] * yfine['F11']) * (yfine['F11'] * Jfine[vars[i]][:,j] -
                    yfine[vars[i]] * Jfine['F11'][:,j])
        else:
            tmatJrat = tmatJ2['F22'][j,:]
            #if j == 0:
            #    tmatJrat = tmatJrat / np.exp(xx_use['lnk'])
            #    all_Jp[:,4*i+2+j] = all_Jp[:,4*i+2+j] / np.exp(xx_use['lnk'])

        truey = yfine[vars[i]] / yfine['F11']
        re_tmatJrat = tmatJrat / truey
        flag = (np.abs(re_tmatJrat) > 1e-3)
        com_tJ = tmatJrat[flag]
        com_pJ = all_Jp[flag,4*i+2+j]/x_std.cpu().numpy()[0][2+j]
        pos = np.logical_and(com_tJ>0.0, com_pJ>0.0 )
        neg = np.logical_and(com_tJ<0.0, com_pJ<0.0 )
        zer = np.logical_and(com_tJ==0.0, com_pJ==0.0 )

        x_data = tmatJrat
        y_data = all_Jp[:,4*i+2+j]/x_std.cpu().numpy()[0][2+j]
        ape = np.abs((y_data - x_data) / x_data) * 100
        s_ape = np.sort(ape)
        sigma_ape = s_ape[int(len(x_data)*0.68)-1]
        print('1 sigma ee = {:.2f}%'.format(sigma_ape), np.where(s_ape > 10)[0])
        s_xdata= np.sort(np.abs(x_data))
        thresh = 1.0e-4 #s_xdata[int(len(x_data)*0.2)-1] # 1.0e-4
        idx0 = np.where(np.abs(x_data) > thresh)[0]
        mre_value = np.nanmean(np.abs(y_data[idx0] - x_data[idx0]) / np.abs(x_data[idx0])) * 100
        print("MRE = {:.2f}%, thresh = {:.4e}".format(mre_value, thresh))
        #pos = np.logical_and(tmatJrat>0.0, all_Jp[:,4*i+2+j]/x_std.cpu().numpy()[0][2+j]>0.0 )
        #neg = np.logical_and(tmatJrat<0.0, all_Jp[:,4*i+2+j]/x_std.cpu().numpy()[0][2+j]<0.0 )
        #zer = np.logical_and(tmatJrat==0.0, all_Jp[:,4*i+2+j]/x_std.cpu().numpy()[0][2+j]==0.0 )

        acc = (len(np.where(pos)[0])+len(np.where(neg)[0])+len(np.where(zer)[0])) / len(np.where(flag)[0])
        print(i,j,'acc: ', acc)
        print(len(np.where(pos)[0]), len(np.where(neg)[0]), len(np.where(zer)[0]) )

        sy_flag = np.where(np.abs(truey) < 1.0e-4)[0]
        sJ_flag = np.where(np.abs(tmatJrat) < 1.0e-4)[0]
        s2_flag = np.where((np.abs(tmatJrat) < 1.0e-4) & (np.abs(truey) < 1.0e-4))[0]
        print('where J and rat are small:', len(sy_flag), sy_flag, len(sJ_flag), sJ_flag, len(s2_flag), s2_flag )
        print('J when rat is small:', truey[np.abs(truey) < 1.0e-6], tmatJrat[np.abs(truey) < 1.0e-6])
        num  = len(np.where(flag)[0])
        if num > 0:
            print('true:', num,len(tmatJrat), tmatJrat[flag], np.max(truey[flag]), np.min(truey[flag]) )
        else:
            print('true:', tmatJrat[flag] )

        fig = plt.figure(figsize=(4.8,3.5))
        ax = fig.add_subplot(111)
        ymi = np.nanmin(np.append(tmatJrat,all_Jp[:,4*i+2+j]/x_std.cpu().numpy()[0][2+j]))
        yma = np.nanmax(np.append(tmatJrat,all_Jp[:,4*i+2+j]/x_std.cpu().numpy()[0][2+j]))
        thresh = 1e-4
#         if yma > 1e-2:
#            thresh = 1e-3
#         else:
#            thresh = 1e-4
        print('thresh:', thresh)
        paths, slope, intercept = scatter(ax, tmatJrat, all_Jp[:,4*i+2+j]/x_std.cpu().numpy()[0][2+j],
                                      color=xx_use[xx[j]],fig=fig,cbar_label=xx[j],cbar_ticks=cb_t[xx[j]],
                                      label_p = 'lower right', thresh=thresh)
        plt.xlim(ymi, yma)
        plt.ylim(ymi, yma)
        ax.set_xlabel(r'$\frac{\partial{'+varsname[i]+'}}{\partial{'+xlab[j]+'}}$ from T-matrix', fontsize=12)
        ax.set_ylabel(r'$\frac{\partial{'+varsname[i]+'}}{\partial{'+xlab[j]+'}}$ from NN', fontsize=12)
        xticks_arr = np.linspace(ymi, yma, 6)
        ax.set_xticks(xticks_arr)
        ax.set_yticks(xticks_arr)
        abyma = max([abs(ymi), abs(yma)])
        if abyma < 1e-4:
           fac = 1e5
           strf= 'e-5'
        elif abyma < 1e-3:
           fac = 1e4
           strf= 'e-4'
        elif abyma < 1e-2:
           fac = 1e3
           strf= 'e-3'
        elif abyma < 0.1:
           fac = 1.0
        ax.set_xticklabels( ['{:.2f}'.format(i*fac) for i in xticks_arr] )
        ax.set_yticklabels( ['{:.2f}'.format(i*fac) for i in xticks_arr] )
        if fac > 1:
           plt.text(ymi,yma+0.01*(yma-ymi), strf)
           plt.text(yma+0.07*(yma-ymi),ymi-0.1*(yma-ymi), strf)
        plt.tight_layout()
        plt.savefig("img/TMlogX_Jac"+xlab[j]+"_scatter_2rat"+vars[i]+"_2y_"+save_name+"_data2.png", dpi=300, facecolor='w', edgecolor='w')
        plt.close()

        print('num:', xx[j], len(np.where(tmatJ2['F22'][j,:] > 1.0e-7)[0]), len(tmatJ2['F22'][j,:]) )

