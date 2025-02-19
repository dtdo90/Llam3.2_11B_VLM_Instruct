import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from huggingface_hub import login
import json
from trl import SFTTrainer
from data import get_custom_dataset, DataCollator
from peft import LoraConfig
from trl import SFTConfig

with open("config.json", "r") as config_file:
    config = json.load(config_file)
    access_token = config["HF_ACCESS_TOKEN"]

login(token=access_token)

def load_model_processor(model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"):
    # bnb_config: nf4 quantization
    bnb_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model=AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        token=access_token)

    processor=AutoProcessor.from_pretrained(model_id,token=access_token)
    return model, processor


def main():
    # load model and processor
    model,processor=load_model_processor()
    # load data
    data=get_custom_dataset()
    ds_train=data["train"]
    ds_test=data["test"]
    data_collator=DataCollator(processor)

    # set up peft and args
    peft_config=LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj"],
        task_type="CAUSAL_LM"
    )
    args=SFTConfig(
        output_dir="llama-3.2-vlm-instruct",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, # number of steps before performing a backward/update pass
        gradient_checkpointing=True,   # gradient checkpointing to save memory
        optim="adamw_torch_fused",
        logging_steps=10,
        learning_rate=2e-4,   # log training loss every 10 steps
        evaluation_strategy="steps", # enable evaluation
        eval_steps=10,        # evaluate evaluation loss every 10 steps
        save_strategy="epoch",
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.3,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="", # dummy field for data collator
        dataset_kwargs={"skip_prepare_dataset": True}
    )
    args.remove_unused_columns=False

    # set up trainer
    trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    data_collator=data_collator,
    peft_config=peft_config,
    tokenizer=processor.tokenizer)

    # explicitly move model to cuda because lora layers are currently in CPU
    trainer.model = trainer.model.cuda()
    
    print("Training start .......")
    trainer.train()

    # save model 
    trainer.save_model(args.output_dir)
    print(f"Training finished. Model saved to {args.output_dir}")

    # free the memory 
    del model
    del trainer
    torch.cuda.empty_cache()
    
if __name__=="__main__":
    main()
