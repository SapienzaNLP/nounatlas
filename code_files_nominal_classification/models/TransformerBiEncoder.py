from typing import Any, Dict, List, Tuple, Union

import lightning.pytorch as pl

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from models.topKAccuracy import top_k_accuracy

import transformers
from transformers import AutoModel, AutoConfig, AutoTokenizer
transformers.utils.logging.set_verbosity_error()

class TransformerBiEncoder(pl.LightningModule):
    def __init__(
            self, 
            # Model hyperparams:
            transformer_name: str = "bert-base-uncased",
            fine_tune_transformer: bool = True,
            mlp_num_hidden: Union[int, List[int]] = None,
            dropout: int = 0.2,
            # Training hyperparams:
            learning_rate: float = 3e-5,
            adam_eps: float = 1e-8,
            adam_weight_decay: float = 0.0,
            accuracy_k: int = 4,            
            seed: int = None,
            *args, **kwargs
        ):
        super().__init__()
        pl.seed_everything(seed)
        
        self.save_hyperparameters()
        self.best_scores = None
        self.validation_step_losses = []
        self.validation_step_accuracies = []
        self.validation_step_f1 = []
        
        self.transformer_name = transformer_name        
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.transformer = AutoModel.from_pretrained(transformer_name, output_hidden_states=True, trust_remote_code=True)

        self.transformer_embedding_size = self.transformer.config.hidden_size
        self.pad_token_id = self.tokenizer(self.tokenizer.pad_token, add_special_tokens = False)["input_ids"][0]

        self.fine_tune = fine_tune_transformer
        if not fine_tune_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
                self.transformer.eval()

        self.dropout = nn.Dropout(dropout)

        # 2) Linear head
        if mlp_num_hidden != None and isinstance(mlp_num_hidden, int) and mlp_num_hidden > 0:
            self.linear_head = nn.Linear(self.transformer_embedding_size, mlp_num_hidden)
        elif mlp_num_hidden != None and isinstance(mlp_num_hidden, list):
            self.linear_head = nn.Sequential(
                nn.Linear(self.transformer_embedding_size, mlp_num_hidden[0]),
                *[nn.Linear(mlp_num_hidden[i],mlp_num_hidden[i+1]) for i in range(len(mlp_num_hidden)-1)]
            )
        else:
            self.linear_head = None

        # 3) Metrics
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.f1 = torchmetrics.F1Score(task="binary")
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings =  torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(sentence_embeddings, p=2, dim=1)
        # return sentence_embeddings

    def forward_embedding(self, text: Union[List[str], List[List[str]]]):   
        
        if isinstance(text, list) and isinstance(text[0], list):
            is_split_into_words = True
        else:
            is_split_into_words = False     

        # 1) Embedding
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            is_split_into_words=is_split_into_words,
            # truncation=True # Warning!
        ).to(self.device)

        tokens_embeddings = self.transformer(**tokens)

        # number of hidden states to consider from the transformer
        n_transformer_hidden_states = 4
        # summing all the considered dimensions
        tokens_embeddings = torch.mean(
            torch.stack(tokens_embeddings.hidden_states[-n_transformer_hidden_states:], dim=0),
            dim=0
        )

        # (batch, sentence_len, transformer_embedding_size) -> (batch, transformer_embedding_size)
        # sentence_embeddings = tokens_embeddings[:,0,:] # Taking the [CLS] embedding as sentence embedding
        sentence_embeddings = self.mean_pooling(tokens_embeddings,tokens["attention_mask"]) # Computing the mean of the valid tokens of the sentence as sentence embedding

        # 2) Linear head
        if self.linear_head != None:
            sentence_embeddings = self.dropout(sentence_embeddings) # -> (batch, transformer_embedding_size)
            sentence_embeddings = self.linear_head(sentence_embeddings) # -> (batch, embedding_size)
        
        return {"tokens_embeddings":tokens_embeddings, "sentence_embeddings":sentence_embeddings} # tokens_embeddings = (batch, sentence_len, embedding_size), sentence_embeddings = (batch, embedding_size)
    
    def forward(self, text: Union[List[str], List[List[str]]]):
        inputs1 = [item[0] for item in text] # (batch, 2, sentence_len) -> (batch, sentence_len) w/ the 1st sentence
        inputs2 = [item[1] for item in text] # (batch, 2, sentence_len) -> (batch, sentence_len) w/ the 2nd sentence
        output1 = self.forward_embedding(inputs1)
        output2 = self.forward_embedding(inputs2)
        sentence_embeddings1 = output1["sentence_embeddings"]
        sentence_embeddings2 = output2["sentence_embeddings"]
        logits = F.cosine_similarity(sentence_embeddings1, sentence_embeddings2, dim=1) # -> (batch)
        return {"sentence_embeddings": [sentence_embeddings1, sentence_embeddings2], "logits": logits}
    
    def loss_function(self, sentence_embeddings1, sentence_embeddings2, binary_targets):
        targets = binary_targets * 2 - 1
        prediction_loss = F.cosine_embedding_loss(sentence_embeddings1, sentence_embeddings2, targets)
        return prediction_loss
    
    
    def loop_step(self, batch: Dict[str, Union[str,torch.Tensor]], stage: str): # batch is what we get from the dataset, with inputs and targets
        inputs, targets = batch["input"], batch["target"]
        outs = self.forward(inputs)
        [sentence_embeddings1, sentence_embeddings2] = outs["sentence_embeddings"]
        loss = self.loss_function(sentence_embeddings1, sentence_embeddings2, targets)
        similarities = outs["logits"]
        preds = (similarities > 0.5).to(torch.long)
        targets = (targets).to(torch.long)
        accuracy = self.accuracy(preds, targets)
        f1 = self.f1(preds,targets)
        
        self.log_dict({f"{stage}_loss": loss, f"{stage}_accuracy": accuracy, f"{stage}_f1": f1}, prog_bar=True, logger=True)

        return loss, accuracy, f1, preds
    
    def training_step(self, batch: Dict[str, Union[str,torch.Tensor]], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, accuracy, f1, _ = self.loop_step(batch, "train")
        return {"loss": loss, "accuracy": accuracy, "F1": f1}

    def on_train_epoch_end(self):
        pass
    
    # def on_train_epoch_start(self):
    #     print(" ")
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, accuracy, f1, _ = self.loop_step(batch, "val")
        self.validation_step_losses.append(loss)
        self.validation_step_accuracies.append(accuracy)
        self.validation_step_f1.append(f1)
        valid_dict = {"val_loss": loss, "val_accuracy": accuracy, "val_f1": f1}
        return valid_dict
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, accuracy, f1, _ = self.loop_step(batch, "test")
        test_dict = {"val_loss": loss, "val_accuracy": accuracy, "val_f1": f1}
        return test_dict
    
    def on_validation_epoch_end(self):
        epoch_loss = torch.stack(self.validation_step_losses).mean().item()
        epoch_accuracy = torch.stack(self.validation_step_accuracies).mean().item()
        epoch_f1 = torch.stack(self.validation_step_f1).mean().item()

        valid_dict = {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy, "val_f1": epoch_f1}

        if self.best_scores is None or self.best_scores["val_accuracy"] < valid_dict["val_accuracy"]:
            self.best_scores = valid_dict
        
        self.validation_step_losses.clear()
        self.validation_step_accuracies.clear()
        self.validation_step_f1.clear()
        print(" ")
        
    def configure_optimizers(self):
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, self.parameters())
        )
        optimizer = optim.Adam(
            trainable_parameters,
            eps=self.hparams.adam_eps,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.adam_weight_decay,
        )
        reduce_lr_on_plateau = ReduceLROnPlateau(
            optimizer, mode="min", verbose=True, min_lr=1e-8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "interval": "epoch",
                "monitor": "train_loss",
                "frequency": 1,
            },
        }
    