import argparse
from typing import Dict, List, Tuple, Optional, Union

from scipy import stats
import torch
from torch import nn
from torch.nn import MSELoss
from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
)

from train import emotion_labels, read_corpus


emotion_categories = ['表出', '経験']


class MultiTaskEmotionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        encodings: Dict[str, List[int]],
        expr_labels: List[List[int]],
        exper_labels: List[List[int]],
        sep_token_id: Optional[int] = 3,
    ) -> None:
        self.encodings = encodings
        self.labels = list(zip(expr_labels, exper_labels))

        self.sep_token_id = sep_token_id

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }

        # token type ids
        sep_mask = (item['input_ids'] == self.sep_token_id).long()
        item['token_type_ids'] = (sep_mask.cumsum(-1) - sep_mask) % 2
                
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self) -> int:
        return len(self.labels)


class BertForMultiTaskSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.num_classifiers = 2
        self.classifiers = nn.ModuleList(
            [nn.Linear(config.hidden_size, config.num_labels) for _ in range(2)]
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        # loss = None
        # if labels is not None:
        #     if self.config.problem_type is None:
        #         if self.num_labels == 1:
        #             self.config.problem_type = "regression"
        #         elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
        #             self.config.problem_type = "single_label_classification"
        #         else:
        #             self.config.problem_type = "multi_label_classification"

        #     if self.config.problem_type == "regression":
        #         loss_fct = MSELoss()
        #         if self.num_labels == 1:
        #             loss = loss_fct(logits.squeeze(), labels.squeeze())
        #         else:
        #             loss = loss_fct(logits, labels)
        #     elif self.config.problem_type == "single_label_classification":
        #         loss_fct = CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     elif self.config.problem_type == "multi_label_classification":
        #         loss_fct = BCEWithLogitsLoss()
        #         loss = loss_fct(logits, labels)

        all_logits = []
        all_loss = []

        for i in range(self.num_classifiers):
            logits = self.classifiers[i](pooled_output)

            loss_fct = MSELoss()
            loss = loss_fct(logits, labels[:, i, :])

            all_logits.append(logits)
            all_loss.append(loss)

        logits = torch.stack(all_logits, 1)
        loss = torch.stack(all_loss).mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    metrics = {}

    for i, emotion_category in enumerate(emotion_categories):
        metrics[emotion_category + '_pearsonr'], _ = stats.pearsonr(
            labels[:, i, :].ravel(),
            logits[:, i, :].ravel(),
        )
        metrics[emotion_category + '_spearmanr'], _ = stats.spearmanr(
            labels[:, i, :].ravel(),
            logits[:, i, :].ravel(),
        )

        for j, emotion_label in enumerate(emotion_labels):
            metrics[emotion_category + '_' + emotion_label + '_pearsonr'], _ = stats.pearsonr(
                labels[:, i, j],
                logits[:, i, j],
            )
            metrics[emotion_category + '_' + emotion_label + '_spearmanr'], _ = stats.spearmanr(
                labels[:, i, j],
                logits[:, i, j],
            )

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_jsonl',
        default='dialogues-train.jsonl'
    )
    parser.add_argument(
        '--val_jsonl',
        default='dialogues-val.jsonl',
    )
    parser.add_argument(
        '--output_dir',
        default='./results',
    )

    parser.add_argument(
        '--pretrained_model_name_or_path',
        # default='nlp-waseda/roberta-base-japanese',
    )
    parser.add_argument(
        '--max_length',
        default=512,
        type=int,
    )

    # parser.add_argument(
    #     '--for_exper',
    #     action='store_true',
    # )
    parser.add_argument(
        '--for_speaker',
        action='store_true',
    )

    parser.add_argument(
        '--per_device_train_batch_size',
        default=16,
        type=int,
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        default=64,
        type=int,
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--num_train_epochs',
        default=3.0,
        type=float,
    )

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
    )

    train_texts, train_expr_labels = read_corpus(
        args.train_jsonl,
        for_exper=False,
        for_speaker=args.for_speaker,
        sep_token=tokenizer.sep_token,
    )
    _, train_exper_labels = read_corpus(
        args.train_jsonl,
        for_exper=True,
        for_speaker=args.for_speaker,
        sep_token=tokenizer.sep_token,
    )
    val_texts, val_expr_labels = read_corpus(
        args.val_jsonl,
        for_exper=False,
        for_speaker=args.for_speaker,
        sep_token=tokenizer.sep_token,
    )
    val_texts, val_exper_labels = read_corpus(
        args.val_jsonl,
        for_exper=True,
        for_speaker=args.for_speaker,
        sep_token=tokenizer.sep_token,
    )

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )

    train_dataset = MultiTaskEmotionDataset(
        train_encodings,
        train_expr_labels,
        train_exper_labels,
        sep_token_id=tokenizer.sep_token_id,
    )
    val_dataset = MultiTaskEmotionDataset(
        val_encodings,
        val_expr_labels,
        val_exper_labels,
        sep_token_id=tokenizer.sep_token_id,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=500,
        logging_strategy='epoch',
        save_strategy='no',
    )

    model = BertForMultiTaskSequenceClassification.from_pretrained(
        args.pretrained_model_name_or_path,
        num_labels=len(emotion_labels),
        problem_type='regression',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    main()
