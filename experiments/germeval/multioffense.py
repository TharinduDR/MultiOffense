from farm.data_handler.data_silo import DataSilo
from farm.data_handler.processor import TextClassificationProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.prediction_head import TextClassificationHead
from farm.modeling.tokenization import Tokenizer
from farm.train import Trainer
from farm.utils import set_all_seeds, initialize_device_settings

set_all_seeds(seed=42)
device, n_gpu = initialize_device_settings(use_cuda=True)
batch_size = 8
lang_model = "bert-base-german-cased"
data_dir_path = "data"
n_epochs = 3
evaluate_every = 100

tokenizer = Tokenizer.load(
    pretrained_model_name_or_path=lang_model,
    do_lower_case=False)

tcp_params = dict(tokenizer=tokenizer,
                  max_seq_len=8,
                  data_dir=data_dir_path,
                  train_filename="GermEval21_Toxic_Train.csv",
                  label_list=[0, 1],
                  metric="f1_macro",
                  dev_filename=None,
                  test_filename=None,
                  dev_split=0.2,
                  delimiter=",",
                  text_column_name="comment_text",
                  label_column_name="Sub1_Toxic")

processor = TextClassificationProcessor(**tcp_params)

data_silo = DataSilo(
    processor=processor,
    batch_size=batch_size)

language_model = LanguageModel.load(lang_model)
prediction_head = TextClassificationHead(num_labels=2, task_name="offense")

model = AdaptiveModel(
    language_model=language_model,
    prediction_heads=[prediction_head],
    embeds_dropout_prob=0.1,
    lm_output_types=["per_sequence"],
    device=device)

model, optimizer, lr_schedule = initialize_optimizer(
    model=model,
    learning_rate=2e-5,
    n_batches=len(data_silo.loaders["train"]),
    n_epochs=1,
    device=device,
    schedule_opts=None)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    data_silo=data_silo,
    epochs=n_epochs,
    n_gpu=n_gpu,
    lr_schedule=lr_schedule,
    evaluate_every=evaluate_every,
    device=device)

trainer.train()
