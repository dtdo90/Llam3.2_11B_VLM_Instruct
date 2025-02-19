import copy 
import itertools
from PIL import Image
import io

import torch
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# Example data
# {
#     "images" = [PIL.Image]
#     "texts" = [
#         {
#             "user": "Question: How many actions are depicted in the diagram?\nChoices:\nA. 6.\nB. 4.\nC. 8.\nD. 7.\nAnswer with the letter.",
#             "assistant": "Answer: D",
#             "source": "TQA"
#         }
#     ]
# }

# check if system header or user header is in a seq
def check_header(targets,seq):
    """ Arguments:
            targets = {system_header, user_header}
            seq = current sequence
    """
    for i in range(len(seq)):
        if seq[i:i+3] in targets:
            return True
    return False

def replace_target(target,seq):
    # replace assistant prompt header <|start_header_id|>assistant<|end_header_id|> by -100
    #          so it won't participate in the loss
    for i in range(len(seq)-3):
        if seq[i:i+3]==target:
            seq[i],seq[i+1],seq[i+2]=-100
        return seq
    
def tokenize_dialogs(dialogs,images,processor):
    """ Tokenize a batch of input data: apply_chat_template convert data to the form
        <|begin_of_text|>
        <|start_header_id|>user<|end_header_id|><|image|>{user_query}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>{user_query}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>{answer}<|eot_id|>
        ...
        <|end_of_text|>
        Arguments:
            dialogs: List of conversations between user and assistant,
                        Each conversation is a list of Question (MCQ) & Answer
            images: List of images, one for each conversation
            processor: tokenize text and image
    """
    # create chat template in the format of llama
    text_prompt=processor.apply_chat_template(dialogs)
    # <|begin_of_text|> is automatically added by processor, so we take it out
    #  Another way: in processor function, set add_special_tokens=False
    text_prompt=[prompt.replace('<|begin_of_text|>','') for prompt in text_prompt]
    # put everything into a batch
    batch=processor(images=images,
                    text=text_prompt,
                    padding=True,
                    return_tensors="pt")
    # store processed tokens: replace the ones not participating in the loss by -100
    label_list=[]
    for i in range(len(batch["input_ids"])):
        # extract tokens from a single dialogue
        dialog_tokens=batch["input_ids"][i].tolist()
        # create a copy so we won't modify the original tokens
        labels=copy.copy(dialog_tokens)
        # find eot_id=128009 which marks the end of a role (user, system, asisstant)
        eot_indices=[i for i,n in enumerate(labels) if n==128009]
        last_idx=0 # last idx a header was found
        # system prompt header "<|start_header_id|>system<|end_header_id|>"
        # and user prompt header "<|start_header_id|>user<|end_header_id|>"
        prompt_header_seqs=[[128006,9125,128007],[128006,882,128007]]
        for _,idx in enumerate(eot_indices):
            # check if prompt header is in curr_seq, if it is, replace everything by -100
            current_seq=labels[last_idx:idx+1]
            if check_header(prompt_header_seqs,current_seq):
                labels[last_idx:idx+1]=[-100]*(idx-last_idx+1)
            else:
                last_idx=idx+1
        # the part that is used in computing loss is "description" in 
        # <|start_header_id|>assistant<|end_header_id|>{description}<|eot_id|><|end_of_text|>
        # Mask the header with -100
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        # Mask the padding token and image token 128256
        for i in range(len(labels)):
            if (
                labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256
            ):  #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch

def get_custom_dataset(data_id="HuggingFaceM4/the_cauldron",split="train",split_ratio=0.9,num_samples=2000):
    """ Load 2000 data on streaming mode
    """
    ds_generator=load_dataset(data_id,name="ocrvqa",split=split,streaming=True)
    ds_iterator=iter(ds_generator)
    

    # collect 2000 samples
    ds_list=[next(ds_iterator) for _ in range(num_samples)]

    # create a Hugging Face dataset class from list
    ds=Dataset.from_list(ds_list)

    # split data into train and test
    ds=ds.train_test_split(test_size=1-split_ratio,shuffle=True,seed=42)
    return ds

class DataCollator:
    """ Define how to batch data according to input image and text"""
    def __init__(self,processor):
        self.processor=processor
        self.processor.tokenizer.padding_size= "right"
    
    def __call__(self, samples):
        """ The call method automatically implements when the class is initiated"""        
        dialogs, images=[], [] 
        for sample in samples:
            image_list,sample_list=sample["images"], sample["texts"]
            if len(image_list)>1:
                raise ValueError("Only support one image per sample")
            
            # convert byte image to PIL and convert to RGB
            image_data=image_list[0]
            image=Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")

            dialog=[] # dialog for 1 single sample
            for sample_dict in sample_list:
                # only add image token in the first sentence
                if not dialog:
                    dialog+=[{"role": "user",
                              "content": [{"type": "image"},
                                          {"type": "text", "text": sample_dict["user"].strip()}]
                            },
                            {"role": "assistant",
                             "content": [{"type": "text",
                                          "text": sample_dict["assistant"].strip()}]
                            },
                    ]
                else:
                    dialog+=[{"role": "user",
                              "content": [{"type": "text"},
                                          {"text": sample_dict["user"].strip()}
                                          ]
                            },
                            {"role": "assistant",
                             "content": [{"type": "text",
                                          "text": sample_dict["assistant"].strip()}]
                            },
                    ]
            dialogs.append(dialog)
            images.append([image])
        return tokenize_dialogs(dialogs,images,self.processor)


# set chat template for processor
# chat_template = """<|begin_of_text|>
# {%- for message in messages %}
# <|start_header_id|>{{ message['role'] }}<|end_header_id|>
# {%- for content in message['content'] %}
# {% if content['type'] == 'image' -%}
# <|image|>
# {% elif content['type'] == 'text' -%}
# {{ content['text'] }}
# {% endif %}
# {%- endfor -%}<|eot_id|>
# {%- endfor -%}<|end_of_text|>"""

# processor.chat_template = chat_template



if __name__=="__main__":
    import json
    from transformers import AutoProcessor
    from huggingface_hub import login
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
    access_token = config["HF_ACCESS_TOKEN"]
    login(token=access_token)

    print("Loading data ...")
    ds=get_custom_dataset(num_samples=10)
    ds_train, ds_test=ds["train"], ds["test"]
    model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"
    processor=AutoProcessor.from_pretrained(model_id,token=access_token)
    collator=DataCollator(processor)
    processed_data=collator.__call__(ds_train)
    
    
