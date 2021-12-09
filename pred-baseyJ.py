import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

task = 1
mlp_layers = 4
mlp_dim = 768 #768
max_epochs = 3000
lr = 0.0065
min_lr = 1e-7

allvars   = ['ext','abs','F11','F22','F33','F44','F12','F34']
dir_name  = 'cabs_rmn_ybaseJac'

# Dubovik's database

data = np.load('./alldata/split_data_'+allvars[task]+'.npz')
train_x = data['train_x']
test_x  = data['test_x']
train_y = data['train_y']
test_y  = data['test_y']
J_mean  = data['J_mean']
J_std   = data['J_std']
print('J_mean', J_mean, J_std)

Usey    = np.vstack((train_y,test_y))
UseX    = np.vstack((train_x,test_x))
num_examples, num_features = UseX.shape
    
#
#
print(UseX.shape, Usey.shape)
print( Usey[Usey[:,-1]>0,:] )


device = 'cuda:2' if torch.cuda.is_available() else 'cpu'


x_mean = train_x.mean(axis=0, keepdims=True)
x_std  = train_x.std(axis=0, keepdims=True)


# Split dataset
train_x, test_x, train_y, test_y = train_test_split(UseX, Usey, test_size=0.2)
    
train_x = (train_x - x_mean) / x_std
test_x = (test_x - x_mean) / x_std

train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y)
test_x = torch.Tensor(test_x)
test_y = torch.Tensor(test_y)
    
# Transform data
x_mean = torch.Tensor(x_mean).cuda(device)
x_std = torch.Tensor(x_std).cuda(device)
J_mean = torch.Tensor(J_mean).cuda(device)
J_std = torch.Tensor(J_std).cuda(device)

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

from skorch.callbacks import Checkpoint, LoadInitState, LRScheduler
cp1 = Checkpoint(dirname='./'+dir_name+'/' )

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
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

train_score = net_regr.score(train_x.cpu(), train_y[:,0].cpu())
test_score = net_regr.score(test_x.cpu(), test_y[:,0].cpu())
print(train_score, test_score)


def get_ut_jacobian(net, x):
    net.to(device)
    batch_size = x.shape[0]
    xt = x 
    xt.requires_grad_(True)
    y = torch.exp(net(xt))
    #print(y.detach().cpu().numpy().shape)
    y.backward(torch.eye(batch_size).cuda(device))
    return y.detach().cpu().numpy(), xt.grad.detach().cpu().numpy(), xt.detach().cpu().numpy()


varsname = ['c_{ext}', 'c_{abs}', 'F_{11}', 'F_{22}']

#######################
# evaluate test dataset
########################

# datapoints not training
# T-matrix data
filename = '../Tmat_Jac/tmat_fine_n_intg.pickle'
with open(filename, 'rb') as f:
    TMdata = pickle.load(f)
X_ori = np.reshape(np.array(TMdata['x']),(int(len(TMdata['x'])/4),4))
y_ori = TMdata['value'][allvars[task]]
J_ori = np.reshape( np.append( TMdata['Jacmi'][allvars[task]], TMdata['Jacmr'][allvars[task]] ), (2,len(TMdata['Jacmr'][allvars[task]])) )
idx   = np.where( (np.abs(X_ori[:,-1]-1.291429) > 1.0e-6 ) & (np.abs(X_ori[:,-1]-1.33) > 1.0e-6 ) )[0]
Xfine    = X_ori[idx,:]
Xfine[:,1] = np.log(Xfine[:,1])
Xfine[:,2] = np.log(Xfine[:,2])
xk    = np.exp(Xfine[:,2])
xn    = Xfine[:,3]
yfine = y_ori[idx]
Jfine = J_ori[:,idx].T

print(Xfine)
Xp = torch.Tensor(Xfine).cuda(device)
Xp = (Xp-x_mean)/x_std
print('Xp',Xp)
with torch.no_grad():
    pred_y = torch.exp(net_regr.module_(Xp)).cpu().numpy()
print('pred_y',pred_y)
from scatter_plot import scatter
fig = plt.figure(figsize=(4.5,3.5))
ax = fig.add_subplot(111)
ymi = np.nanmin(np.append(yfine,pred_y))
yma = np.nanmax(np.append(yfine,pred_y))
paths, slope, intercept = scatter(ax, yfine, pred_y, color='b',label_p = 'lower right',fsize=12)
plt.xlim(ymi, yma)
plt.ylim(ymi, yma)
ax.set_xlabel(r'T-matrix $'+varsname[task]+'$', fontsize=12) #+r' ($\mu$m$^{-1}$)'
ax.set_ylabel(r'NN Predict $'+varsname[task]+'$', fontsize=12)
xticks_arr = np.linspace(ymi, yma, 6)
ax.set_xticks(xticks_arr)
ax.set_xticklabels( ['{:.2f}'.format(i*1e3) for i in xticks_arr] )
ax.set_yticks(xticks_arr)
ax.set_yticklabels( ['{:.2f}'.format(i*1e3) for i in xticks_arr] )
plt.text(ymi,yma+0.01*(yma-ymi), 'e-3')
plt.text(yma+0.01*(yma-ymi),ymi, 'e-3')
plt.tight_layout()
plt.savefig("img/TMlogX_scatter_"+allvars[task]+"_rmn_ybaseJac_data2.png", dpi=300, facecolor='w', edgecolor='w')
plt.close()

from torch.utils.data import DataLoader, TensorDataset
pred_X_t = torch.tensor(Xfine).float()
loader = DataLoader( pred_X_t, batch_size=5000)
all_yp = np.array([])
all_Jp = np.array([])
for data in loader:
    Xp = data.cuda(device)
    Xp = (Xp-x_mean)/x_std
    pred_y, pred_J, xt = get_ut_jacobian(net_regr.module_, Xp)
    #print(pred_y.shape, pred_J.shape)
    if len(all_yp) == 0:
        all_yp = pred_y
        all_Jp = pred_J
    else:    
        all_yp = np.vstack((all_yp, pred_y))
        all_Jp = np.vstack((all_Jp, pred_J))
    print(all_yp.shape, all_Jp.shape)

from scatter_plot import scatter
xx = ['lnk','n']
xx_use = {}
cb_t   = {}
xx_use['lnk'] = Xfine[:,2]
xx_use['n']   = Xfine[:,3]
cb_t['lnk']   = xx_use['lnk'][:181*14*5:181*14]
cb_t['n']     = xx_use['n'][::181*14*5]
for j in range(2):
    fig = plt.figure(figsize=(4.8,3.5))
    ax = fig.add_subplot(111)
    
    pos = np.logical_and(Jfine[:,j]>0.0, all_Jp[:,2+j]/x_std.cpu().numpy()[0][2+j]>0.0 )
    neg = np.logical_and(Jfine[:,j]<0.0, all_Jp[:,2+j]/x_std.cpu().numpy()[0][2+j]<0.0 )
    zer = np.logical_and(Jfine[:,j]==0.0, all_Jp[:,2+j]/x_std.cpu().numpy()[0][2+j]==0.0 )
        
    acc = (len(np.where(pos)[0])+len(np.where(neg)[0])+len(np.where(zer)[0])) / len(Jfine[:,j])
    print(j,'acc: ', acc)
    
    ymi = np.nanmin(np.append(Jfine[:,j],all_Jp[:,2+j]/x_std.cpu().numpy()[0][2+j]))
    yma = np.nanmax(np.append(Jfine[:,j],all_Jp[:,2+j]/x_std.cpu().numpy()[0][2+j]))
    paths, slope, intercept = scatter(ax, Jfine[:,j], all_Jp[:,2+j]/x_std.cpu().numpy()[0][2+j], 
                                      color=xx_use[xx[j]],fig=fig,cbar_label=xx[j],cbar_ticks=cb_t[xx[j]],
                                      label_p = 'lower right')
    plt.xlim(ymi, yma)
    plt.ylim(ymi, yma)
    ax.set_xlabel(r'$\frac{\partial{'+varsname[task]+'}}{\partial{'+xx[j]+'}}$ from T-matrix', fontsize=12)
    ax.set_ylabel(r'$\frac{\partial{'+varsname[task]+'}}{\partial{'+xx[j]+'}}$ from NN', fontsize=12)
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
    ax.set_xticklabels( ['{:.2f}'.format(i*fac) for i in xticks_arr] )
    ax.set_yticklabels( ['{:.2f}'.format(i*fac) for i in xticks_arr] )
    plt.text(ymi,yma+0.01*(yma-ymi), strf)
    plt.text(yma+0.07*(yma-ymi),ymi-0.1*(yma-ymi), strf)
    plt.tight_layout()
    plt.savefig("img/TMlogX_Jac"+xx[j]+"_scatter_"+allvars[task]+"_rmn_ybaseJac_data2.png", dpi=300, facecolor='w', edgecolor='w')
    plt.close()


