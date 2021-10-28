from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

import torch

set_all_seeds(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
lang_model = "bert-base-german-cased"
data_dir_path = "experiments/germeval/data"
n_epochs = 3
evaluate_every = 100

tokenizer = Tokenizer.load(
    pretrained_model_name_or_path=lang_model,
    do_lower_case=False)

tcp_params = dict(tokenizer=tokenizer,
                  max_seq_len=128,
                  data_dir=data_dir_path,
                  train_filename="GermEval21_Toxic_Train.csv",
                  quote_char='"',
                  dev_filename=None,
                  test_filename=None,
                  dev_split=0.2,
                  delimiter=","
                  )

processor = TextClassificationProcessor(**tcp_params)

processor.add_task(name="toxic", task_type="classification", label_list=["0", "1"], metric="f1_macro",
                   text_column_name="comment_text",
                   label_column_name="Sub1_Toxic")
processor.add_task(name="engaging", task_type="classification", label_list=["0", "1"], metric="f1_macro",
                   text_column_name="comment_text",
                   label_column_name="Sub2_Engaging")
processor.add_task(name="fact_claim", task_type="classification", label_list=["0", "1"], metric="f1_macro",
                   text_column_name="comment_text",
                   label_column_name="Sub3_FactClaiming")

data_silo = DataSilo(
    processor=processor,
    batch_size=batch_size, max_processes=1)

language_model = LanguageModel.load(lang_model)
toxic_head = TextClassificationHead(num_labels=2, task_name="toxic")
engage_head = TextClassificationHead(num_labels=2, task_name="engaging")
claim_head = TextClassificationHead(num_labels=2, task_name="fact_claim")

model = AdaptiveModel(
    language_model=language_model,
    prediction_heads=[toxic_head, engage_head, claim_head],
    embeds_dropout_prob=0.1,
    lm_output_types=["per_sequence", "per_sequence", "per_sequence"],
    device=device)

model, optimizer, lr_schedule = initialize_optimizer(
    model=model,
    learning_rate=2e-5,
    n_batches=len(data_silo.loaders["train"]),
    n_epochs=n_epochs,
    device=device,
    schedule_opts=None)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=n_epochs,
    n_gpu=1,
    lr_schedule=lr_schedule,
    evaluate_every=evaluate_every,
    device=device)

trainer.train()
