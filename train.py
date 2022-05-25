import argparse
import json
from typing import Dict, List, Tuple, Optional

from scipy import stats
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


emotion_labels = ['怒り', '期待', '喜び', '信頼', '恐れ', '驚き', '悲しみ', '嫌悪']


def read_corpus(
    jsonl: str,
    for_exper: Optional[bool] = False,
    for_speaker: Optional[bool] = False,
    use_mrph: Optional[bool] = True,
    sep_token: Optional[bool] = '[SEP]',
) -> Tuple[List[str], List[List[int]]]:
    texts = []
    labels = []

    with open(jsonl) as f:
        for line in f:
            line_dict = json.loads(line)

            ctx = []

            if for_speaker:
                if for_exper:
                    del line_dict[-1]

                else:
                    if use_mrph:
                        mrph_list = line_dict[0]['utt']['mrph_list']
                        text = ' '.join(mrph_list)

                    else:
                        text = line_dict[0]['utt']['text']

                    ctx.append(text)
                    del line_dict[0]

            for u_dict in line_dict:
                utterance = u_dict['utt']

                if use_mrph:
                    mrph_list = utterance['mrph_list']
                    text = ' '.join(mrph_list)

                else:
                    text = utterance['text']

                ctx.append(text)
                ctx_text = sep_token.join(ctx)

                emotion = u_dict['emo']
                emotion_category = 'exper' if for_exper else 'expr'

                label = []
                for emotion_label in emotion_labels:
                    emotion_value = 0
                    for emotion_intensity in ['strong', 'weak']:
                        if emotion_label in emotion[emotion_category][emotion_intensity]:
                            emotion_value += 1

                    label.append(emotion_value)

                texts.append(ctx_text)
                labels.append(label)

    return texts, labels


class EmotionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        encodings: Dict[str, List[int]],
        labels: List[List[int]],
        sep_token_id: Optional[int] = 3,
    ) -> None:
        self.encodings = encodings
        self.labels = labels

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


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    metrics = {}

    metrics['pearsonr'], _ = stats.pearsonr(labels.ravel(), logits.ravel())
    metrics['spearmanr'], _ = stats.spearmanr(labels.ravel(), logits.ravel())

    for i, emotion_label in enumerate(emotion_labels):
        metrics[emotion_label + '_pearsonr'], _ = stats.pearsonr(
            labels[:, i],
            logits[:, i],
        )
        metrics[emotion_label + '_spearmanr'], _ = stats.spearmanr(
            labels[:, i],
            logits[:, i],
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
        default='nlp-waseda/roberta-base-japanese',
    )
    parser.add_argument(
        '--max_length',
        default=512,
        type=int,
    )

    parser.add_argument(
        '--for_exper',
        action='store_true',
    )
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
    )

    train_texts, train_labels = read_corpus(
        args.train_jsonl,
        for_exper=args.for_exper,
        for_speaker=args.for_speaker,
        sep_token=tokenizer.sep_token,
    )
    val_texts, val_labels = read_corpus(
        args.val_jsonl,
        for_exper=args.for_exper,
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

    train_dataset = EmotionDataset(
        train_encodings,
        train_labels,
        sep_token_id=tokenizer.sep_token_id,
    )
    val_dataset = EmotionDataset(
        val_encodings,
        val_labels,
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

    model = AutoModelForSequenceClassification.from_pretrained(
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
