import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sklearn.metrics as sklm
import pandas as pd
cxr_labels = ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion','Lung Opacity', 'No Finding','Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def multi_label_accuracy(output, target, thresholds=[0.5]*14):
    # output = output.detach().cpu().numpy()
    with torch.no_grad():
        output = torch.sigmoid(output).detach().cpu().numpy() 
        ## save a copy of output
        output_copy = output.copy()
        ## round based on thesholds
        for i in range(len(thresholds)):
            output[:,i] = (output[:,i] >= thresholds[i]).astype(int)
        pred_S = np.round(output) 
        target = target.detach().cpu().numpy()
        gt_S  = np.asarray(target) 
        # pred_S = np.round(output)      #will round to the nearest even number
        # pred_S = output
        acc =  accuracy_score(gt_S,pred_S)

        # aucs = []
        # if output.shape[0]>128:
            
        #     for i in range(len(thresholds)):
        #         # sklm.roc_auc_score(gt_S[:,i], output_copy[:,i])
        #         ## calculate average AUC
        #         auc = sklm.roc_auc_score(gt_S[:,i], output_copy[:,i])
        #         print('AUC for label {} is {}'.format(i, auc))
        #         aucs.append(auc)

        # print('Average AUC is {}'.format(np.mean(aucs)))
        # f1m = f1_score(gt_S,pred_S,average = 'macro', zero_division=1)
        # f1mi = f1_score(gt_S,pred_S,average = 'micro', zero_division=1)
        # print('f1_Macro_Score{}'.format(f1m))
        # print('f1_Micro_Score{}'.format(f1mi))
        # print('Accuracy{}'.format(acc))
    return acc

def auc(output, target, thresholds):
        # output = output.detach().cpu().numpy()
    with torch.no_grad():
        output = torch.sigmoid(output).detach().cpu().numpy() 
        ## save a copy of output
        output_copy = output.copy()
        ## round based on thesholds
        for i in range(len(thresholds)):
            output[:,i] = (output[:,i] >= thresholds[i]).astype(int)
        pred_S = np.round(output) 
        target = target.detach().cpu().numpy()
        gt_S  = np.asarray(target) 
        # pred_S = np.round(output)      #will round to the nearest even number
        # pred_S = output
        # acc =  accuracy_score(gt_S,pred_S
        flatten_acc = sklm.accuracy_score(gt_S.flatten(), pred_S.flatten())

        precision = sklm.precision_score(gt_S, pred_S, average='micro')
        recall = sklm.recall_score(gt_S, pred_S, average='micro')
        f1 = 2 * precision * recall / (precision + recall)

        aucs = []
        if output.shape[0]>128:
            
            for i in range(len(thresholds)):
                # sklm.roc_auc_score(gt_S[:,i], output_copy[:,i])
                ## calculate average AUC
                auc = sklm.roc_auc_score(gt_S[:,i], output_copy[:,i])
                print('AUC for label {} is {}'.format(i, auc))
                aucs.append(auc)

        print('Average AUC is {}'.format(np.mean(aucs)))
        # f1m = f1_score(gt_S,pred_S,average = 'macro', zero_division=1)
        # f1mi = f1_score(gt_S,pred_S,average = 'micro', zero_division=1)
        # print('f1_Macro_Score{}'.format(f1m))
        # print('f1_Micro_Score{}'.format(f1mi))
        # print('Accuracy{}'.format(acc))
        return aucs, precision, recall, f1, flatten_acc


def auc_threshold(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output).detach().cpu().numpy() 
        # output = np.round(output) 
        target = target.detach().cpu().numpy()
        gt_S  = np.asarray(target) 
        # pred_S = output
        # pred_S = np.round(output)   

        f1s = []
        thres = []
        
        for column in range(gt_S.shape[1]):
            p, r, t = sklm.precision_recall_curve(gt_S[:,column], output[:,column])
            # Choose the best threshold based on the highest F1 measure
            f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
            bestthr = t[np.where(f1 == max(f1))]
            print('best threshold for label {} is {}'.format(column, bestthr))
            f1s.append(max(f1))
            thres.append(bestthr[0])   
    return thres, f1s


def macro_f1(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output).detach().cpu().numpy() 
        output = np.round(output) 
        target = target.detach().cpu().numpy()
        gt_S  = np.asarray(target) 
        pred_S = np.round(output)   
        f1m = f1_score(gt_S,pred_S, average = 'macro', zero_division=1)
    return f1m

def micro_f1(output, target):
    with torch.no_grad():
        output = torch.sigmoid(output).detach().cpu().numpy() 
        output = np.round(output) 
        target = target.detach().cpu().numpy()
        gt_S  = np.asarray(target) 
        pred_S = np.round(output)   
        f1mi = f1_score(gt_S,pred_S, average = 'micro', zero_division=1)
    return f1mi 


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)



## chexbert acc

### a function calculating the f1 score from chexbert labels
def bootstrap_acc():
    dir = '/net/scratch/chacha/future_of_work/cache/'
    labels_csv = 'testing_matching_chexbert.csv'

    # labels_csv
    bootstrap_dir ='/net/scratch/chacha/future_of_work/bootstrap_test/'

    # true_labels = pd.read_csv(args.bootstrap_dir + 'labels.csv.gz', compression='gzip').fillna(0)[useful_labels]
    true_labels = pd.read_csv(bootstrap_dir + 'labels_test_split_chacha.csv').fillna(0)
    # [useful_labels]
    # true_labels = pd.read_csv(bootstrap_dir + 'labels_brent.csv').fillna(0)[useful_labels]

    pred_labels = pd.read_csv(dir + labels_csv).fillna(0)

    pred_csv = pd.read_csv('/net/scratch/chacha/future_of_work/report-generation/saved/DenseChexpertModel/lr_1e-5_dense121_ddp_stepLR_20/test_matching_gt.csv')
    ## merge with pred_csv based on study id and subject id
    pred_labels['study_id'] = pred_csv['study_id']
    pred_labels['subject_id'] = pred_csv['subject_id']


    pred_labels['subject_id'] = pred_labels['subject_id'].fillna(0)
    ## astype
    pred_labels['subject_id'] = pred_labels['subject_id'].astype(int)
    pred_labels['study_id'] = pred_labels['study_id'].astype(int)
    ## filter from true_labels with same study id and subject id as the in pred_labels
    true_labels_merged = true_labels.merge(pred_labels[['study_id','subject_id']], on=['study_id','subject_id'], how='inner')
    
    ## same study id means same label? 
    true_labels_no_duplicates = true_labels.drop_duplicates(subset=['study_id'] + cxr_labels, inplace = False)
    # = true_labels[['study_id'] + cxr_labels].drop_duplicates(inplace=False)
    pred_labels_no_duplicates = pred_labels.drop_duplicates(subset=['study_id'] + cxr_labels, inplace = False)

    pred_labels_no_duplicates = pred_labels_no_duplicates.merge(true_labels_no_duplicates, on=['study_id'], how='left')
    ## calculate accuracy based on 14 labels



    # [useful_labels]
    # pred_labels = pd.read_csv(dir + 'labeled_reports.csv').fillna(0)[useful_labels]
    x_labels = [x+'_x' for x in cxr_labels]
    y_labels = [x+'_y' for x in cxr_labels]
    np_true_labels = pred_labels_no_duplicates[y_labels].to_numpy()
    np_pred_labels = pred_labels_no_duplicates[x_labels].to_numpy()
    np_pred_labels[np_pred_labels == -1] = 0
    np_true_labels[np_true_labels == -1] = 0
    opts = np.array([0,1])
    assert np.all(np.isin(np_pred_labels, opts)) # make sure all 0s and 1s


    f1_s = f1_score(np_true_labels, np_pred_labels, average='macro')

    print(f1_s)
    # scores = []
    # for i in range(10): # 10 bootstrap
    #     indices = np.loadtxt(bootstrap_dir + str(i) + '/indices.txt', dtype=int)
    #     batch_score = f1_batch(indices, np_pred_labels, np_true_labels)
    #     scores.append(batch_score)
    ## what is that for?
    # interval = st.t.interval(0.95, df=len(scores)-1, loc=np.mean(scores), scale=st.sem(scores))
    # mean = sum(scores) / len(scores)
    # print(f'f1 mean: {round(mean, 3)}, plus or minus {round(mean - interval[0], 3)}')

def f1_batch(indices, pred_labels, true_labels):
    f1_macro = f1_score(true_labels[indices,:], pred_labels[indices,:], average='macro')
    return f1_macro

# bootstrap_acc()
