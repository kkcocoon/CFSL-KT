import os
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import random
from torch.utils.data.dataset import Dataset
import torch
from sklearn.cluster import AgglomerativeClustering

# In[hyperspectral dataset]:
def loadData(name, num_components=None, data_path=None, half=True):
   if data_path==None:
      data_path = os.path.join(os.getcwd(),'data')
        
   if name == 'IP':
      data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
      labels = sio.loadmat(os.path.join(data_path, 'indian_pines_gt.mat'))['indian_pines_gt']
   elif name == 'IP_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected_EMAP.mat'))['indian_pines_corrected']
      labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
   elif name == 'IP9_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected_EMAP.mat'))['indian_pines_corrected']
      labels = sio.loadmat(os.path.join(data_path, 'indian9_gt.mat'))['gt']
   elif name == 'IP9':
      data = sio.loadmat(os.path.join(data_path, 'indian_pines_corrected.mat'))['indian_pines_corrected']
      labels = sio.loadmat(os.path.join(data_path, 'indian9_gt.mat'))['gt']
   elif name == 'SV':
      data = sio.loadmat(os.path.join(data_path, 'salinas_corrected.mat'))['salinas_corrected']
      labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
   elif name == 'SV_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'salinas_corrected_EMAP.mat'))['salinas_corrected']
      labels = sio.loadmat(os.path.join(data_path, 'salinas_gt.mat'))['salinas_gt']
   elif name == 'PU':
      data = sio.loadmat(os.path.join(data_path, 'paviaU.mat'))['paviaU']
      labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
   elif name == 'PU_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'paviaU_EMAP.mat'))['paviaU']
      labels = sio.loadmat(os.path.join(data_path, 'paviaU_gt.mat'))['paviaU_gt']
   elif name == 'KSC':
      data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
      labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
   elif name == 'Chikusei_EMAP':
      data = sio.loadmat(os.path.join(data_path, 'Chikusei_EMAP.mat'))['chikusei']
      labels = sio.loadmat(os.path.join(data_path, 'Chikusei14_gt.mat'))['chikusei_gt']
   elif name == 'Houston_EMAP':
      data = sio.loadmat(os.path.join(data_path,'Houston_EMAP.mat'))['Houston']
      labels = sio.loadmat(os.path.join(data_path,'Houston_gt.mat'))['Houston_gt']
   else:
      print("NO DATASET")
      exit()
      
   shapeor = data.shape
      
   data = data.reshape(-1, data.shape[-1])
   if num_components != None:
      data = PCA(n_components=num_components).fit_transform(data)
      shapeor = np.array(shapeor)
      shapeor[-1] = num_components
   #data = MinMaxScaler().fit_transform(data)  
   data = StandardScaler().fit_transform(data)  # X = (X-X_mean)/X_std
   data = data.reshape(shapeor)
   num_class = len(np.unique(labels)) - 1
   labels = labels.astype(np.uint16)
   print(labels.shape)
   return data, labels, num_class

class HyperData(Dataset):
    def __init__(self, dataset):
        self.data = dataset[0].astype(np.float32)
        self.labels = []
        for n in dataset[1]: self.labels += [int(n)]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels


def load_hyper(args):
   data, gt, numclass = loadData(args.dataset, num_components=args.components,data_path=args.data_path)
   pixels,labels,idx,idy = createImageCubes(data, gt, patchsize=args.patchsize, removeZeroLabels=True,ridx=True)
   print(max(idx))
   print('idy={}'.format(len(idy)))
   bands = pixels.shape[-1]; numberofclass = len(np.unique(labels))
   shape = pixels.shape[0:2];
   if args.tr_percent < 1: # split by percent
      x_train, x_test, y_train, y_test, idx_te, idy_te = split_data(pixels, labels,args.tr_percent, idx=idx,idy=idy,rand_state=args.rand_state)
   else: # split by samples per class
      x_train, x_test, y_train, y_test, idx_te, idy_te = split_data_fix(pixels, labels, args.tr_percent, idx=idx,idy=idy,rand_state=args.rand_state)
   if args.use_val: 
      if args.val_percent<1: # noted: x_test0 and x_test
         x_val, x_test0, y_val, y_test0, _, _ = split_data(x_test, y_test, args.val_percent, idx=idx,idy=idy,rand_state=args.rand_state)
      else:
         x_val, x_test0, y_val, y_test0, _, _ = split_data_fix(x_test, y_test, args.val_percent, idx=idx,idy=idy,rand_state=args.rand_state)
   del pixels
   train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"),y_train))
   
   test_hyper = None
   if args.use_test: test_hyper  = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"),y_test))
   
   val_hyper = None
   if args.use_val: val_hyper = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"),y_val))
   
   #kwargs = {'num_workers': 1, 'pin_memory': True}
   kwargs = {'num_workers': 0, 'pin_memory': True}
   
   if args.tr_bsize>len(x_train):
      args.tr_bsize=len(x_train)
   train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=True, drop_last=True,**kwargs)
   
   if args.use_test: 
      test_loader  = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
   else:
      test_loader = None
      
   if args.use_val:
      val_loader  = torch.utils.data.DataLoader(val_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
   else:
      val_loader  = None
   return train_loader, test_loader, val_loader, numberofclass, bands, shape, idx_te, idy_te, gt



#def load_hyper_all(args):
#   data, labels, numclass = loadData(args.dataset, num_components=args.components)
#   pixels,labels,idx,idy = createImageCubes(data, labels, patchsize=args.patchsize, removeZeroLabels=False,ridx=True)
#   all_hyper = HyperData((np.transpose(pixels, (0, 3, 1, 2)).astype("float32"),labels))
#   kwargs = {'num_workers': 0, 'pin_memory': True}
#   all_loader = torch.utils.data.DataLoader(all_hyper, batch_size=args.tr_bsize, shuffle=False, drop_last=False,**kwargs)
#   return all_loader

# In[randomly select data]:
   
def random_unison(a,b, rstate=None):
   assert len(a) == len(b)
   p = np.random.RandomState(seed=rstate).permutation(len(a))
   return a[p], b[p]


def split_data_fix(pixels, labels, n_samples, idx=None,idy=None,rand_state=None):
   train_set_size = [n_samples] * len(np.unique(labels))
   return split_data(pixels, labels, 0, train_set_size, idx,idy,rand_state)

def split_data(pixels, labels, percent, train_set_size=None, idx=None,idy=None,rand_state=None):
   
   pixels_number = np.unique(labels, return_counts=1)[1]
   # print("np.unique(labels, return_counts=1)",np.unique(labels, return_counts=1))
   # print("pixels_number",np.unique(labels, return_counts=1)[1])

   if train_set_size is None or len(train_set_size)!=len(pixels_number):
      train_set_size = [int(np.ceil(a*percent)) for a in pixels_number]

   #print("train_set_size",train_set_size)
   # Prealloc memory, faster
   tr_size = int(sum(train_set_size)) # 0
   te_size = int(sum(pixels_number)) # 0
   
   sizetr = np.array([tr_size]+list(pixels.shape)[1:])
   sizete = np.array([te_size]+list(pixels.shape)[1:])
   train_x = np.empty((sizetr)); train_y = np.empty((tr_size),dtype=np.int16); test_x = np.empty((sizete)); test_y = np.empty((te_size),dtype=np.int16)
   idx_te = np.empty((sizete[0]))
   idy_te = np.empty((sizete[0]))

   tr_count = 0
   tt_count = 0
   for cl in np.unique(labels):
      
      bind = np.where(labels==cl)[0]
      pixels_cl = pixels[bind]
      labels_cl = labels[bind]
      
      # If there are not enough samples for class cl, take 75% for training, without using the setting 'n_samples'.
      tlen = len(bind)
      if tlen<train_set_size[cl]:
         train_set_size[cl] = int(0.75*tlen)
         
      pind = np.random.RandomState(seed=rand_state).permutation(tlen)
      trind = pind[0:train_set_size[cl]] # random select samples
      ttind = pind[train_set_size[cl]:]
 
      train_x[tr_count:tr_count+len(trind)] = pixels_cl[trind]
      train_y[tr_count:tr_count+len(trind)] = labels_cl[trind]
      tr_count = tr_count+len(trind)
      test_x[tt_count:tt_count+len(ttind)] = pixels_cl[ttind]
      test_y[tt_count:tt_count+len(ttind)] = labels_cl[ttind]
      idx_te[tt_count:tt_count+len(ttind)] = idx[bind[ttind]]
      idy_te[tt_count:tt_count+len(ttind)] = idy[bind[ttind]]

      tt_count = tt_count+len(ttind)
      # print(bind[pind[0:5]])
      #print(trind)
   
   train_x = train_x[0:tr_count]
   train_y = train_y[0:tr_count]
   test_x = test_x[0:tt_count]
   test_y = test_y[0:tt_count]
   idx_te = idx_te[0:tt_count]
   idy_te = idy_te[0:tt_count]

   # train_x, train_y = random_unison(train_x, train_y, rstate=rand_state)
   return train_x, test_x, train_y, test_y, idx_te, idy_te


# In[create cubes]:

def createImageCubes(X, y, patchsize=11, removeZeroLabels=True, half=False, ridx=True, step=1):
   if removeZeroLabels:
      (tidx,tidy) = np.where(y>0)  # 0 means background
   else:
      (tidx,tidy) = np.where(y>-1)

   print("tidx", tidx)

   print("tidy", tidy)

   print("tidx and tidy are the coordinates of non-zero pixels ")
    
#   # if there are too much samples, take half of samples for experiment
#   if len(tidx)>20000 and half: 
#      np.random.seed(123);
#      rind = np.random.permutation(len(tidx))
#      rind = rind[:len(rind)//2]
#      tidx = tidx[rind]
#      tidy = tidy[rind]
      
   margin = int((patchsize - 1) / 2)
   dsize = len(np.arange(0,patchsize,step)) # downsample by step
   X_pad = np.lib.pad(X,((margin,margin),(margin,margin),(0,0)),'symmetric') # 对称边界延拓，而不是补0
   patchesData = np.zeros((len(tidx), dsize, dsize, X.shape[2]))
   patchesLabels = np.zeros((len(tidx),))
   tidx = tidx + margin
   tidy = tidy + margin
   pi=0
   for r,c in zip(tidx,tidy):
      patch = X_pad[r-margin:r+margin+1:step, c-margin:c+margin+1:step]   
      patchesData[pi, :, :, :] = patch
      patchesLabels[pi] = y[r-margin, c-margin]
      pi=pi+1
   
   patchesLabels = patchesLabels.astype(np.int32)-1  # without background, labels start from 0
    
#   if removeZeroLabels:
#      rind = np.random.permutation(len(patchesLabels))
#      patchesData = patchesData[rind]
#      patchesLabels = patchesLabels[rind].astype("int")
#      tidx = tidx[rind]
#      tidy = tidy[rind]
   
   idx = tidx - margin
   idy = tidy - margin
   
   print('max(idx) = ')
   print(max(idx))
   if ridx:
      return patchesData, patchesLabels,idx,idy
   else:
      return patchesData, patchesLabels

# In[]:
      
   
def predict(testloader, model, domain='target'):
   model.eval()
   preds = []
   labels = []
   feas=[]
   for batch_idx, (test_x, test_y) in enumerate(testloader):
      test_x = test_x.cuda()
      fea, pred = model(test_x,domain)
      [preds.append(a) for a in pred.data.cpu().numpy()] 
      [labels.append(a) for a in test_y] 
      [feas.append(a.detach().cpu().numpy()) for a in fea] 
   feas = np.array(feas)
   preds_mx = np.array(preds)
   preds = np.argmax(preds_mx, axis=1)
   return preds, preds_mx, np.array(labels),feas

def accuracy(output, target, topk=(1,)):
   """Computes the precision@k for the specified values of k"""
   maxk = max(topk)
   batch_size = target.size(0)

   _, pred = output.topk(maxk, 1, True, True)
   pred = pred.t()
   correct = pred.eq(target.view(1, -1).expand_as(pred))

   res = []
   for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
   return res


def AA_andEachClassAccuracy(confusion_matrix):
   #counter = confusion_matrix.shape[0]
   list_diag = np.diag(confusion_matrix)
   list_raw_sum = np.sum(confusion_matrix, axis=1)
   each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
   average_acc = np.mean(each_acc)
   return each_acc, average_acc


def reports(y_pred, y_test):
   classification = classification_report(y_test, y_pred)
   oa = accuracy_score(y_test, y_pred)
   confusion = confusion_matrix(y_test, y_pred)
   each_acc, aa = AA_andEachClassAccuracy(confusion)
   kappa = cohen_kappa_score(y_test, y_pred)

   return classification, confusion, list(np.round(np.array([oa, aa, kappa] + list(each_acc)) * 100, 2))

# In[t-SNE visualize]:
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# We import seaborn to make nice plots.
# import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})

def plot_TSNE(X,y,cc=None, picname=None):
#   tsne_obj = TSNE(random_state=2020).fit(X)
#   digits_proj = tsne_obj.transform(X)
   digits_proj = TSNE(random_state=2020).fit_transform(X)
   f, ax, sc, txts = scatter(digits_proj, y)
   plt.scatter
   if cc:
      xx = digits_proj[-cc:,0]
      yy = digits_proj[-cc:,1]
      ax.scatter(xx,yy,c='r',marker='o')
      
      for i in range(cc):
         ax.text(xx[i], yy[i], str(i+1), fontsize=20, c='r')
   plt.show()

   if picname:
      plt.savefig(picname, dpi=120)
   

def scatter(x, y):
    NumClass = max(y)+1    
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", NumClass))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[y.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(NumClass):
        # Position of each label.
        xtext, ytext = np.median(x[y == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i+1), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def same_seeds(seed):
   torch.manual_seed(seed)
   if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
   np.random.seed(seed)  # Numpy module.
   random.seed(seed)  # Python random module.
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

#from sklearn.datasets import load_digits
#digits = load_digits()
## We first reorder the data points according to the handwritten numbers.
#X = np.vstack([digits.data[digits.target==i]
#               for i in range(10)])
#y = np.hstack([digits.target[digits.target==i]
#               for i in range(10)])
#plot_TSNE(X,y)   