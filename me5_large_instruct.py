from transformers import TrainerCallback
import huggingface_hub
import wandb
import argparse
import torch
import shutil
import os

torch.cuda.empty_cache()

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

def get_args():
    parser = argparse.ArgumentParser(description="Tokens for login")
    parser.add_argument("--hf_token", type=str, required=True, help="HF TOKEN")
    parser.add_argument("--wandb_api_key", type=str, required=True, help="WANDB API KEY")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size of the training")
    parser.add_argument("--epoch", type=int, default=5, help="Epoch of the training")
    return parser.parse_args()

args = get_args()

huggingface_hub.login(token=args.hf_token)
wandb.login(key=args.wandb_api_key)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer(
    'Maminirina1/multilingual-e5-large-instruct-mg-en-v1'
).to(device)
model.gradient_checkpointing_enable()

ds = load_dataset("Maminirina1/MalagasyEnglish")

# train_dataset = ds["train"]
# eval_dataset = ds["validation"]
train_dataset = ds["train"].shuffle(seed=42).select(range(1000))
eval_dataset = ds["validation"].shuffle(seed=42).select(range(200))
    
class PushAndCleanCallback(TrainerCallback):
    def __init__(self, trainer, delete_checkpoints=True):
        """
        Custom callback to push model to Hugging Face Hub and clean local checkpoints.
        
        Args:
            trainer: The Trainer instance.
            delete_checkpoints (bool): Whether to delete local checkpoint directories.
        """
        self.trainer = trainer
        self.delete_checkpoints = delete_checkpoints

    def on_save(self, args, state, control, **kwargs):
        print(f"Manual push to Hub at step {state.global_step}")
        
        # Push model using the trainer instance
        self.trainer.push_to_hub(commit_message=f"Checkpoint at step {state.global_step}")
        
        # Delete checkpoint files
        if self.delete_checkpoints:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
                print(f"Deleted local checkpoint: {checkpoint_dir}")

        # Optionally stop further saves
        control.should_save = False
        return control


loss = MultipleNegativesRankingLoss(model=model)

args = SentenceTransformerTrainingArguments(
    output_dir="./multilingual-e5-large-instruct-mg-en-v1",
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    bf16=False,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    push_to_hub=True,
    hub_model_id="Maminirina1/multilingual-e5-large-instruct-mg-en-v1",
    hub_token=args.hf_token,
    logging_steps=1000,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=8,
    run_name="multilingual-e5-large-instruct-mg-en-v1",
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)
callback = PushAndCleanCallback(trainer=trainer, delete_checkpoints=True)
trainer.add_callback(callback)
trainer.train()