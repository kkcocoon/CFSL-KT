import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import argparse
import datetime

from models import MyNetwork, DomainClassifier, RandomLayer, weights_init
from models import KernalTripletHardLoss

import time
import numpy as np
import scipy.io as sio
import auxil

# In[parser]:

parser = argparse.ArgumentParser(description='main_CFSLKT')
args_main = parser.parse_args()    
  
#def main(args_main):

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--episodes', default=50, type=int, help='(2000) Number of total episodes to run')
parser.add_argument("-f","--feature_dim",type = int, default = 256)
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='(0.001) Important! Initial learning rate')

#parser.add_argument('--data_path', default=r'E:\6.PythonPro\10.Group Experiment\datasets-EMAPorg', type=str, help='data path')
# parser.add_argument('--data_path', default=r'/opt/data/private/data_HSI', type=str, help='data path')
parser.add_argument('--data_path', default=r'G:\data_HSI', type=str, help='data path')

parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
parser.add_argument('--dataset', default='IP_EMAP', type=str, help='dataset (options: SV_EMAP, PU_EMAP, IP9_EMAP, IP, PU, SV, KSC, Houston_EMAP)')
parser.add_argument('--tr_percent', default=5, type=float, help='(100 or 0.05) Samples of train set')
parser.add_argument('--tr_bsize', default=128, type=int, help='(400) Important! Mini-batch train size')
parser.add_argument('--te_bsize', default=128, type=int, help='(1000) Mini-batch test size')
parser.add_argument('--use_test',default=True, type=bool, help='Use test set')
parser.add_argument('--use_val',default=True, type=bool, help='Use validation set')
parser.add_argument('--val_percent', default=0.1, type=float, help='(0.05) samples of val set')
parser.add_argument('--patchsize',  default=9, type=int, help='spatial patch size')
parser.add_argument('--optimizer',  default='SGD', type=str, help='optimizer')

parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='(1e-4) Weight decay ')

parser.add_argument('--loss_use_kernel', default=True, type=float, help='loss_use_kernel')
parser.add_argument('--loss_hard_weight', default=5, type=float, help='Important! Weight for triplet loss')
parser.add_argument('--loss_hard_margin', default=2.0, type=bool, help='Margin for triplet hard loss')
parser.add_argument('--loss_domain_weight', default=0.01, type=float, help='loss_domain_weight')
parser.add_argument('--loss_source_weight', default=1, type=float, help='loss_source_weight')

parser.add_argument('--net', default='MyNetwork', type=str, help='(pRestNet, MyNetwork) Net model')
parser.add_argument('--rand_state', default=1331, type=int, help='(None,123) Random seed')

# Ensure the results for every random experiment are the same for the same random seed.
auxil.same_seeds(0)

args = parser.parse_args()

args_dict = vars(args)
args_main_dict = vars(args_main)
for key in args_main_dict.keys():
   args_dict[key] = args_main_dict[key]
args = argparse.Namespace(**args_dict)

for k in args_main_dict.keys():
   print(k,':', args_dict[k])

#for key in args_dict.keys():
#   print(key,':', args_dict[key])


# In[load data]:

source_band_num=51   # only for init, useless
source_class_num=14  # only for init, useless
print("\n******Load source******")
args_source = parser.parse_args()
args_source.dataset='Chikusei_EMAP'
args_source.tr_percent = 200 
args_source.tr_bsize = 256
args_source.use_test = False
args_source.use_val = False
source_loader, _ , _ , source_class_num, source_band_num, source_shape, _, _,_ = auxil.load_hyper(args_source)
print('Source: {}, Class: {}, Train: {}'.format(args_source.dataset,source_class_num,source_loader.dataset.__len__()))


print("\n******Load target******")
train_loader, test_loader, val_loader, target_class_num, target_band_num, target_shape, idx_te, idy_te, gt = auxil.load_hyper(args)
print('Target: {}, Class: {}, Train: {}, Test: {}\n'.format(args.dataset, target_class_num, train_loader.dataset.__len__(),test_loader.dataset.__len__()))

print("Training episodes = {}\n".format(args.episodes))

# In[create network]:
feature_encoder = MyNetwork(source_band_num, target_band_num, source_class_num, target_class_num, args.feature_dim)
domain_classifier = DomainClassifier()

# In[train]:

criterion_hard = KernalTripletHardLoss(margin=args.loss_hard_margin,use_kernal=args.loss_use_kernel).cuda()
criterion_CE = torch.nn.CrossEntropyLoss().cuda()
criterion_domain = torch.nn.BCEWithLogitsLoss().cuda()


if args.optimizer == 'Adam':
   feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.lr)
   domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=args.lr)
else:
   feature_encoder_optim = torch.optim.SGD(feature_encoder.parameters(), lr=args.lr, momentum=args.momentum,
                  weight_decay=args.weight_decay, nesterov=True)
   domain_classifier_optim = torch.optim.SGD(domain_classifier.parameters(), lr=args.lr, momentum=args.momentum,
                  weight_decay=args.weight_decay, nesterov=True)


feature_encoder.cuda()
domain_classifier.cuda()

source_iter = iter(source_loader)   # many labeled source samples for training
domain_iter = iter(test_loader)     # many unlabeled target samples for training
train_iter = iter(train_loader)     # few labeled target samples for training

start_time=datetime.datetime.now()

s_acc=0
train_loss = []
best_acc = -1
best_episode = 0
best_results = []
A = np.zeros([10, target_class_num])

loss_domain = torch.Tensor([0]).cuda()
loss_domainT = torch.Tensor([0]).cuda()
loss_hard_source = torch.Tensor([0]).cuda()
loss_CE_source = torch.Tensor([0]).cuda()

for episode in range(args.episodes):

   '''get data from source domain and target domain'''
      
   try:
      source_data, source_label = source_iter.next()
   except Exception as err:
      source_iter = iter(source_loader)
      source_data, source_label = source_iter.next()
   try:
      domain_data, _ = domain_iter.next()
   except Exception as err:
      domain_iter = iter(test_loader)
      domain_data, _ = domain_iter.next()
   try:
      train_data, train_label = train_iter.next()
   except Exception as err:
      train_iter = iter(train_loader)
      train_data, train_label = train_iter.next()
      
   domain_classifier.train()
   feature_encoder.train()
   
   # ----------------------------------------------------------------------------

   '''get feature output'''
   source_features, source_outputs = feature_encoder(source_data.cuda(), domain='source')
   domain_features,  domain_outputs = feature_encoder(domain_data.cuda(), domain='target')
   train_features, train_outputs = feature_encoder(train_data.cuda(), domain='target')

   softmax_output_source = torch.nn.Softmax(dim=1)(source_outputs)
   # softmax_output_domain = torch.nn.Softmax(dim=1)(domain_outputs)
   softmax_output_train = torch.nn.Softmax(dim=1)(train_outputs)

   # ----------------------------------------------------------------------------
   '''train domain_classifier'''
   #  to better discriminant source and target
   
   domain_logits_source = domain_classifier(source_features, episode)
   domain_logits_train = domain_classifier(train_features, episode)
   domain_logits_domain = domain_classifier(domain_features, episode)

   domain_logits = torch.cat([domain_logits_source, domain_logits_train, domain_logits_domain], dim=0)

   domain_label = torch.zeros([source_data.shape[0] + train_data.shape[0] + domain_data.shape[0], 1]).cuda()
   domain_label[:source_data.shape[0]] = 1  # source:1, target:0


   loss_domain = criterion_domain(domain_logits, domain_label)

   # Do not consider domain loss when episode<=1000
   if episode>1000:
     domain_classifier.zero_grad()
     loss_domain.backward(retain_graph=True)
     domain_classifier_optim.step()

   # ----------------------------------------------------------------------------
   '''train feature_encoder'''
   # to better align source and target, and classify samples simultaneously

   loss_hard_source = criterion_hard(source_features, source_label.cuda())
   loss_CE_source = criterion_CE(source_outputs, source_label.cuda())
   loss_hard_train = criterion_hard(train_features, train_label.cuda())
   loss_CE_train = criterion_CE(train_outputs, train_label.cuda())
  
   domain_logits_source = domain_classifier(source_features, episode)
   domain_logits_train = domain_classifier(train_features, episode)
   domain_logits_domain = domain_classifier(domain_features, episode)

   domain_logits = torch.cat([domain_logits_source, domain_logits_train, domain_logits_domain], dim=0)
   
   # Note that here labels of target are set to 1. 
   domain_label = torch.zeros([source_data.shape[0] + train_data.shape[0] + domain_data.shape[0], 1]).cuda()
   domain_label[:] = 1  # source:1, target:1.

   loss_domainT = criterion_domain(domain_logits, domain_label)

   # target loss
   loss = loss_CE_train
   loss = loss + args.loss_hard_weight * loss_hard_train

   # Do not consider source loss when episode<=500
   if episode>500:
     loss = loss + args.loss_source_weight*(loss_CE_source +  args.loss_hard_weight * loss_hard_source)

   # Do not consider domain loss when episode<=1000
   if episode>1000:   #and episode<3000:
     loss = loss + args.loss_domain_weight * loss_domainT


   if episode<1000 or episode>1100:
      feature_encoder.zero_grad()
      loss.backward()
      feature_encoder_optim.step()

   '''record results'''
   if (episode + 1) % 10 == 0:
      feature_encoder.eval()

      train_loss.append(loss.item())
      s_acc = auxil.accuracy(softmax_output_source, source_label.cuda())[0].item()

      tr_acc = auxil.accuracy(softmax_output_train, train_label.cuda())[0].item()

      print('e:{}, domain: {:.2f}, {:.2f} hard: {:.4f}, {:.4f}, CE: {:.4f}, {:.4f}, total: {:.2f}, s_acc: {:.2f}, tr_acc:{:.2f}'.format(
              episode + 1,
              loss_domain.item(),
              loss_domainT.item(),
              loss_hard_source.item(),
              loss_hard_train.item(),
              loss_CE_source.item(),
              loss_CE_train.item(),
              loss.item(),
              s_acc,
              tr_acc))

   if (episode<100 and (episode+1)%10==0) or (episode+1)%50==0:

      if args.use_val:
         print("Verifying ...")
         preds,preds_mx,labels,feas = auxil.predict(val_loader, feature_encoder)
      else:
         print("Testing ...")
         preds,preds_mx,labels,feas = auxil.predict(test_loader, feature_encoder)
      
      classification, confusion, results = auxil.reports(preds,labels)
      test_acc = results[0]

      if test_acc >= best_acc:
         state = {
               'args_dict': args_dict,
               'best_acc': test_acc,
               'state_dict': feature_encoder.state_dict(),
               'optimizer' : feature_encoder_optim.state_dict(),
         }
         torch.save(state, "best_model.pth")
         best_episode = episode
         best_acc = test_acc
         best_results = results
         A = np.diag(confusion) / np.sum(confusion, 1, dtype=np.float)

      print('episode:{}, acc = {:.2f}'.format(episode + 1, test_acc))
      print('best_episode:{}, best_acc = {:.2f}'.format(best_episode + 1, best_acc))


duration_tr = datetime.datetime.now()-start_time

checkpoint = torch.load("best_model.pth")
feature_encoder.load_state_dict(checkpoint['state_dict'])
preds,preds_mx,labels,feas = auxil.predict(test_loader, feature_encoder)
classification, confusion, best_results = auxil.reports(preds,labels)

fname = 'train/{}_{}_{}_{}'.format(args.net,args.dataset,args.tr_percent,args.rand_state)
fname = fname + '_hardw_{}-domainw_{}-episode_{}'.format(args.loss_hard_weight,args.loss_domain_weight,best_episode+1)

gt = gt+1
results_test = {'pred': preds, 'test_label': labels, 'idx': idx_te, 'idy': idy_te, 'gt':gt}
fname = fname + '_{:.2f}'.format(best_results[0])
np.savez(fname, results_test)


for ij in np.arange(0, len(preds)):
   x = int(idx_te[ij])
   y = int(idy_te[ij])
   gt[x, y] = preds[ij]+1
data = {
   'data': gt
}
sio.savemat(fname+'.mat', data)

sresult = '{:.2f} # {}-trtime_{}'.format(best_results[0], fname, duration_tr.seconds)
file_handle = open('result_SemiDA.txt','a')
file_handle.write('\n')
file_handle.writelines(sresult)
file_handle.close()
print(sresult)

#   return best_results,  A
#
#if __name__ == '__main__':
#	main(args_main)