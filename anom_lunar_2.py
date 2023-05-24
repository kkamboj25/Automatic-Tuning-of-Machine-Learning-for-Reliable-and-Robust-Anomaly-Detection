import numpy as np
import time
#load the data file, extract training (x) and testing (tx,ty) samples
f=np.load("/global/cardio.npz")
x,tx,ty=f["x"],f["tx"],f["ty"]

#import the model we want to fit (an Isolation Forest) and the evaluation metric (ROC-AUC)
from pyod.models.lunar import LUNAR  ## Kartik Comment
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from flaml import tune
##from flaml import tune
def train_one(**hyper):
        clf=LUNAR(**hyper)
        clf.fit(x)
        p=clf.decision_function(tx)
        auc =  roc_auc_score(ty,p)
        return auc



def optimization(config: dict):
        t0=time.time()
        auc=train_one(**config)
        t1=time.time()

        return {'score':auc, 'evaluation_cost':t1-t0}




hyparam={'model_type':tune.choice(['WEIGHT','SCORE']),'negative_sampling': tune.choice(['UNIFORM','SUBSPACE','MIXED']),'n_neighbours':tune.randint(lower=1,upper=10),'epsilon':tune.uniform(0.0,0.1),'n_epochs':tune.randint(lower=100,upper=1000),'lr': tune.loguniform(lower=0.001, upper=1.0),'wd':tune.choice([0.1]),'proportion':tune.randint(lower=1,upper=10)}#,'scaler':tune.choice(['StandardScaler','MinMaxScaler'])}


sol=tune.run(optimization, metric='score', mode='max', config=hyparam,resources_per_trial={'cpu':4,'gpu':0}, num_samples=100, time_budget_s=3600)
print(sol.best_trial.last_result)
#Create an instance of the model and fit it to the training data
#clf=LUNAR()#n_components=10,covariance_type='full',tol=0.001,reg_covar=1e-06,max_iter=100,n_init=1,init_params='kmeans',weights_init=None,means_init=None, precisions_init=None,random_state=None, warm_start=False)
#clf.fit(x)

#Predict the anomaly scores for the testing data
#p=clf.decision_function(tx)

#Evaluate the model using the ROC-AUC metric
#auc=roc_auc_score(ty,p)
#print(auc)
