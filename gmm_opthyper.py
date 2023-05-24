import numpy as np
import time
import json
#load the data file, extract training (x) and testing (tx,ty) samples
from pyod.models.gmm import GMM  ## Kartik Comment
from sklearn.metrics import roc_auc_score
from flaml import tune

import os
pth="/global/datasets/"
files=[pth+"/"+fil for fil in os.listdir(pth) if os.path.isfile(pth+"/"+fil)]

def train_one(**hyper):
    try:
        clf=GMM(**hyper)
        clf.fit(x)
        p=clf.decision_function(tx)
        auc =  roc_auc_score(ty,p)
    except Exception:
        auc=0
    return auc



def optimization(config: dict):
        t0=time.time()
        auc=train_one(**config)
        t1=time.time()

        return {'score':auc, 'evaluation_cost':t1-t0}
list1=[]

for i in files:
        f=np.load(i)
        x,tx,ty=f["x"],f["tx"],f["ty"]
        hyparam={'n_components':tune.randint(lower=1,upper=8),'covariance_type':tune.choice(["full", "tied", "diag", "spherical"]),'max_iter':tune.randint(lower=1,upper=100),'reg_covar':1e-3}
   # except ValueError:
        sol=tune.run(optimization, metric='score', mode='max', config=hyparam,resources_per_trial={'cpu':4,'gpu':0}, num_samples=50, time_budget_s=60)
        list1.append(sol.best_result) 
    
        with open("data.json","w") as f:
            #dump the dictionary into the file
            json.dump(list1,f,indent=2)

