from Lightning.dataloader import OpusDataModule
from Lightning.model_lightning import BilangLightning

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from config import get_config

config_ = get_config()

def runner(train=True):
    data = OpusDataModule()
    data.setup()

    src, tgt = data.tokenizer_src, data.tokenizer_tgt
    model = BilangLightning(learning_rate=1e-4, tokenizer_src=src, tokenizer_tgt=tgt)

    trainer = pl.Trainer(precision=16, max_epochs=config_['num_epochs'], accelerator="gpu")

    trainer.fit(model, data)
