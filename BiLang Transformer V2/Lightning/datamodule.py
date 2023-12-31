from pathlib import Path
import os
import torch

from tokenizers import Tokenizer
from datasets import load_dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import BilingualDataset
from utils import get_or_build_tokenizer
from config import get_config
config_ = get_config()

class OpusDataModule(LightningDataModule):
    def __init__(self, config=config_):
        super().__init__()
        
        self.config = config
        self.train_data = None
        self.val_data = None
        self.tokenizer_src = None
        self.tokenizer_tgt = None

    def prepare_data(self):
        load_dataset(
            "opus_books", 
            f"{self.config['lang_src']}-{self.config['lang_tgt']}", 
            split="train"
        )

    def setup(self, stage=None):
        if not self.train_data and not self.val_data:
            ds_raw = load_dataset(
                "opus_books", 
                f"{self.config['lang_src']}-{self.config['lang_tgt']}", 
                split="train"
            )

            # Build tokenizers
            self.tokenizer_src = get_or_build_tokenizer(self.config, ds_raw, self.config["lang_src"])
            self.tokenizer_tgt = get_or_build_tokenizer(self.config, ds_raw, self.config["lang_tgt"])

            ds = []

            for item in ds_raw:
                src_target_pair = item
                src_text = src_target_pair['translation'][self.config['lang_src']]
                tgt_text = src_target_pair['translation'][self.config['lang_tgt']]
                
                enc_input_tokens = self.tokenizer_src.encode(src_text).ids

                if len(enc_input_tokens) > 150:
                    continue

                dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
                if len(dec_input_tokens) > len(enc_input_tokens) + 10:
                    continue

                ds.append(item)

            # keep 90% for training, 10% for validation
            train_ds_size = int(0.9 * len(ds))
            val_ds_size = len(ds) - train_ds_size
            train_ds_raw, val_ds_raw = random_split(ds, [train_ds_size, val_ds_size])

            print(f"Length of filtered dataset : {len(ds)}")
            print(f"Train DS Size : {len(train_ds_raw)}")
            print(f"  Val DS Size : {len(val_ds_raw)}")

            self.train_data = BilingualDataset(
                train_ds_raw,
                self.tokenizer_src,
                self.tokenizer_tgt,
                self.config["lang_src"],
                self.config["lang_tgt"],
                self.config["seq_len"],
            )

            self.val_data = BilingualDataset(
                val_ds_raw,
                self.tokenizer_src,
                self.tokenizer_tgt,
                self.config["lang_src"],
                self.config["lang_tgt"],
                self.config["seq_len"],
            )

            # Find the max length of each sentence in the source and target sentnece
            max_len_src = 0
            max_len_tgt = 0

            for item in ds:
                src_ids = self.tokenizer_src.encode(item["translation"][self.config["lang_src"]]).ids
                tgt_ids = self.tokenizer_tgt.encode(item["translation"][self.config["lang_tgt"]]).ids
                max_len_src = max(max_len_src, len(src_ids))
                max_len_tgt = max(max_len_tgt, len(tgt_ids))

            print(f"Max length of source sentence: {max_len_src}")
            print(f"Max length of target sentence: {max_len_tgt}")

            print(f"Source Vocab Size : {self.tokenizer_src.get_vocab_size()}")
            print(f"Target Vocab Size : {self.tokenizer_tgt.get_vocab_size()}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data, 
            batch_size=1, 
            shuffle=False
        )
    
def collate_fn(batch):
    encoder_input_max = max(x['encoder_str_length'] for x in batch)
    decoder_input_max = max(x['decoder_str_length'] for x in batch)

    encoder_input = []
    decoder_input = []
    encoder_mask = []
    decoder_mask = []
    label = []
    src_text = []
    tgt_text = []

    for b in batch:
        encoder_input.append(b['encoder_input'][:encoder_input_max])
        decoder_input.append(b['decoder_input'][:decoder_input_max])
        encoder_mask.append((b['encoder_mask'][0, 0, :encoder_input_max]).unsqueeze(0).unsqueeze(0).unsqueeze(0))
        decoder_mask.append((b['decoder_mask'][0, :decoder_input_max, :decoder_input_max]).unsqueeze(0).unsqueeze(0))
        label.append(b['label'][:decoder_input_max])
        src_text.append(b['src_text'] )
        tgt_text.append(b['tgt_text'] )

    return {
        "encoder_input": torch.vstack(encoder_input),
        "decoder_input": torch.vstack(decoder_input),
        "label": torch.vstack(label),
        "encoder_mask": torch.vstack(encoder_mask),
        "decoder_mask": torch.vstack(decoder_mask),
        "src_text": src_text,
        "tgt_text": tgt_text
    }