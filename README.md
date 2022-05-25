# Building a Dialogue Corpus Annotated with Expressed and Experienced Emotions

The repository for the paper [Building a Dialogue Corpus Annotated with Expressed and Experienced Emotions](https://aclanthology.org/2022.acl-srw.3/) at ACL SRW 2022.

## Requirements

- Python==3.9.4

## Training

The code for training is based on the document [Fine-tuning with custom datasets](https://huggingface.co/transformers/v4.12.5/custom_datasets.html).

```bash
python train.py \
    --train_jsonl $TRAIN_JSONL \
    --val_jsonl $VAL_JSONL \
    --output_dir $OUTPUT_DIR \
    --pretrained_model_name_or_path $PRETRAINED_MODEL_NAME_OR_PATH
```

Run the model for prediction:

```bash
python pred.py \
    --test_jsonl $TEST_JSONL \
    --model_path $OUTPUT_DIR
```

## Citation

```bibtex
@inproceedings{ide-kawahara-2022-building,
    title = "Building a Dialogue Corpus Annotated with Expressed and Experienced Emotions",
    author = "Ide, Tatsuya  and
      Kawahara, Daisuke",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-srw.3",
    pages = "21--30",
    abstract = "In communication, a human would recognize the emotion of an interlocutor and respond with an appropriate emotion, such as empathy and comfort. Toward developing a dialogue system with such a human-like ability, we propose a method to build a dialogue corpus annotated with two kinds of emotions. We collect dialogues from Twitter and annotate each utterance with the emotion that a speaker put into the utterance (expressed emotion) and the emotion that a listener felt after listening to the utterance (experienced emotion). We built a dialogue corpus in Japanese using this method, and its statistical analysis revealed the differences between expressed and experienced emotions. We conducted experiments on recognition of the two kinds of emotions. The experimental results indicated the difficulty in recognizing experienced emotions and the effectiveness of multi-task learning of the two kinds of emotions. We hope that the constructed corpus will facilitate the study on emotion recognition in a dialogue and emotion-aware dialogue response generation.",
}
```
