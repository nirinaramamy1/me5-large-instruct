from transformers import TrainerCallback
import huggingface_hub
import wandb
import argparse
import torch
import shutil
import os
import logging

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
    def __init__(self, trainer, delete_checkpoints=True, hub_push=True):
        """
        Custom callback to push model to Hugging Face Hub and optionally clean local checkpoints.
        
        Args:
            trainer: The Trainer instance.
            delete_checkpoints (bool): Whether to delete local checkpoint directories.
            hub_push (bool): Whether to push to Hugging Face Hub.
        """
        self.trainer = trainer
        self.delete_checkpoints = delete_checkpoints
        self.hub_push = hub_push
        self.logger = logging.getLogger(__name__)

    def on_save(self, args, state, control, **kwargs):
        """
        Called when the Trainer saves a checkpoint.
        
        Args:
            args: Training arguments.
            state: Training state.
            control: Training control object.
            **kwargs: Additional arguments.
        """
        self.logger.info(f"Manual push to Hub at step {state.global_step}")

        if self.hub_push:
            try:
                # Push model to Hub
                self.trainer.push_to_hub(commit_message=f"Checkpoint at step {state.global_step}")
                self.logger.info(f"Successfully pushed checkpoint at step {state.global_step} to Hub")
            except Exception as e:
                self.logger.error(f"Failed to push to Hub: {str(e)}")

        return control

    def on_step_end(self, args, state, control, **kwargs):
        """
        Called at the end of each training step. Delete checkpoint after Trainer is done with it.
        """
        if self.delete_checkpoints and state.global_step % args.save_steps == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            try:
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
                    self.logger.info(f"Deleted local checkpoint: {checkpoint_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to delete checkpoint {checkpoint_dir}: {str(e)}")

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
callback = PushAndCleanCallback(trainer=trainer, delete_checkpoints=True, hub_push=True)
trainer.add_callback(callback)
trainer.train()