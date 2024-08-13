import time, os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import numpy as np
import torch
DEVICE = torch.device("cuda:1")
import warnings
warnings.filterwarnings('ignore')
from gensim.models import KeyedVectors
from datetime import timedelta
from B_loaddata import *
torch.cuda.manual_seed()


model_name = 'C_vgg'
meta = 'both'
lang,model_mode = model_name.split('_')
is_mld = True
is_inter = True
is_intra = True
cat_or_add = 'add'
share_or_single = 'share'

is_baseline = False

if is_baseline:
    from B_model_old import *
else:
    from B_model_inter_intra import *

from B_train import *


class_sentiment = 7
class_intention = 5
class_offensiveness = 4
class_m_occurrence = 2
class_m_category = 2

train_data, val_data, test_data = build_dataset(lang=lang,mode=model_mode,meta=meta)

train_iter = build_iterator(train_data, batch_size=48, device=DEVICE,lang=lang)
val_iter = build_iterator(val_data, batch_size=48, device=DEVICE,lang=lang)
test_iter = build_iterator(test_data, batch_size=48, device=DEVICE,lang=lang)
model = mergeNet(DEVICE,class_sentiment,class_intention,
                 class_offensiveness,class_m_occurrence,class_m_category,
                 is_mld,is_inter,is_intra,cat_or_add,model_mode,share_or_single)
train(model, train_iter, val_iter, test_iter,class_sentiment,class_intention,class_offensiveness,class_m_occurrence,class_m_category,meta)