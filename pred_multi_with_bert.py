import argparse

from transformers import (
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from train_multi_with_bert import (
    emotion_labels,
    read_corpus,
    MultiTaskEmotionDataset,
    BertForMultiTaskSequenceClassification,
    compute_metrics,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_jsonl',
        default='dialogues-test.jsonl'
    )
    parser.add_argument(
        '--model_path',
        default='./results'
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
        '--max_length',
        default=512,
        type=int,
    )
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(
        args.model_path,
    )

    test_texts, test_expr_labels = read_corpus(
        args.test_jsonl,
        for_exper=False,
        for_speaker=args.for_speaker,
        sep_token=tokenizer.sep_token,
    )
    _, test_exper_labels = read_corpus(
        args.test_jsonl,
        for_exper=True,
        for_speaker=args.for_speaker,
        sep_token=tokenizer.sep_token,
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )
    test_dataset = MultiTaskEmotionDataset(
        test_encodings,
        test_expr_labels,
        test_exper_labels,
        sep_token_id=tokenizer.sep_token_id,
    )

    training_args = TrainingArguments(
        output_dir=args.model_path,
        # evaluation_strategy='epoch',
        # per_device_train_batch_size=args.per_device_train_batch_size,
        # per_device_eval_batch_size=args.per_device_eval_batch_size,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        # weight_decay=0.01,
        # num_train_epochs=args.num_train_epochs,
        # warmup_steps=500,
        # logging_strategy='epoch',
        # save_strategy='no',
    )

    model = BertForMultiTaskSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=len(emotion_labels),
        problem_type='regression',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        # tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print(trainer.predict(test_dataset).metrics)


if __name__ == '__main__':
    main()
