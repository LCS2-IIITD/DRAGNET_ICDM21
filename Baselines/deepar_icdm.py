import os
import warnings
from tqdm import tqdm
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gc

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch

from pytorch_forecasting import Baseline, NBeats, TimeSeriesDataSet, DeepAR, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import SMAPE, QuantileLoss, RMSE, DistributionLoss
import math

thread_store = pickle.load(open('./time_series.pkl', 'rb'))

x = 0
y = 2760

df = pd.DataFrame(columns=['series', 'time_idx', 'value'])
val_df = pd.DataFrame(columns=['series', 'time_idx', 'value'])
final = []
means_stds = []
val_final = []
groups = []
val_groups = []
file_names = []
val_file_names = []
time_indices = []
val_time_indices = []
final_list = []
pl.seed_everything(42)
fails = []
for i in tqdm(range(x,y)):
        try:
            if len(thread_store[i][0] > 25):
                means_stds.append([thread_store[i][0].mean(), thread_store[i][0].std()])
                norm = (thread_store[i][0] - thread_store[i][0].mean())/thread_store[i][0].std()
                final += list(norm)[:int(0.8*len(norm))]
                val_final += list(norm)[int(0.8*len(norm)):]
                groups += [i for j in norm[:int(0.8*len(norm))]]
                val_groups += [i for j in norm[int(0.8*len(norm)):]]
                time_indices += [j for j in range(len(list(norm)[:int(0.8*len(norm))]))]
                val_time_indices += [j for j in range(len(list(norm)[int(0.8*len(norm)):]))]
                file_names.append(thread_store[i][5]) 
                val_file_names.append(thread_store[i][5]) 
        except:
            pass

df['series'] = groups
df['time_idx'] = time_indices
df['value'] = final

val_df['series'] = val_groups
val_df['time_idx'] = val_time_indices
val_df['value'] = val_final

training = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="value",
    group_ids=["series"],
    time_varying_unknown_reals=["value"],
    max_encoder_length=25,
    max_prediction_length=1,
)
validation = TimeSeriesDataSet(
    val_df,
    time_idx="time_idx",
    target="value",
    group_ids=["series"],
    time_varying_unknown_reals=["value"],
    max_encoder_length=25,
    max_prediction_length=1,
)

batch_size = 256
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

trainer = pl.Trainer(gpus=1, gradient_clip_val=0.1)
net = DeepAR.from_dataset(
    training, 
    learning_rate=3e-2, 
    weight_decay=1e-2,
    time_varying_reals_encoder=["value"],
    time_varying_reals_decoder = []
)

res = trainer.tuner.lr_find(
    net, 
    train_dataloader=train_dataloader, 
    val_dataloaders=val_dataloader, 
    min_lr=1e-5,
    )

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=25,
    gpus=1,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=30,
)


net = DeepAR.from_dataset(
    training,
    learning_rate=4e-3,
    log_interval=10,
    log_val_interval=1,
    weight_decay=1e-2,
    time_varying_reals_encoder=["value"],
    time_varying_reals_decoder = []
)

net.hparams.learning_rate = res.suggestion()

trainer.fit(
    net,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)

best_model_path = trainer.checkpoint_callback.best_model_path
best_model = DeepAR.load_from_checkpoint(best_model_path)

rmse_vals = 0

for i in tqdm(range(2760, len(thread_store))):
    try:
        if len(thread_store[i][0] > 25):
            means = thread_store[i][0].mean()
            stds = thread_store[i][0].std()
            norm = (thread_store[i][0] - means)/stds
            groups = [i for j in norm]
            time_indices = [j for j in range(len(list(norm)))]
            test_df = pd.DataFrame(columns=['series', 'time_idx', 'value'])
            test_df['series'] = groups
            test_df['time_idx'] = time_indices
            test_df['value'] = norm
            testing = TimeSeriesDataSet(
                test_df,
                time_idx="time_idx",
                target="value",
                group_ids=["series"],
                time_varying_unknown_reals=["value"],
                max_encoder_length=25,
                min_prediction_length=len(thread_store[i][0]) - 25,
                max_prediction_length=len(thread_store[i][0]) - 25
            )
            test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
            predictions = best_model.predict(test_dataloader)*stds
            pickle.dump([norm[25:]*stds + means, predictions.numpy().flatten()*stds + means], open('deepar_preds_fountana/'+ thread_store[i][5] + '.pkl', 'wb'))
    except:
        pass