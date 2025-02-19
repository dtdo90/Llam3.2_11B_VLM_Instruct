import torch
import requests
import json
import io


from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import login

with open("config.json", "r") as config_file:
    config = json.load(config_file)
    access_token = config["HF_ACCESS_TOKEN"]

login(token=access_token)

# [{'role': 'user',
#   'content': [{'type': 'image'},
#    {'type': 'text',
#     'text': 'Who wrote this book?\nProvide a short and direct response.'}]},
#  {'role': 'assistant',
#   'content': [{'type': 'text', 'text': 'Heather Thomas.'}]},
#  {'role': 'user',
#   'content': [{'type': 'text'},
#    {'text': 'What is the title of this book?\nProvide a short and direct response.'}]},
#  {'role': 'assistant',
#   'content': [{'type': 'text',
#     'text': "The Vegetarian Society's New Vegetarian Cookbook."}]},
#     ...
# ]


def load_fine_tune_model(base_model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",adapter_path="llama-3.2-vlm-instruct"):
    processor=AutoProcessor.from_pretrained(base_model_id)
    model=AutoModelForVision2Seq.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    # create a deep copy 
    model.load_adapter(adapter_path)

    return model, processor


def generate(model,image,query,processor,dialog=[]):
    # every time the user input a new query, the dialog is updated and the model compute news output
    if not dialog: # append image token only to the 1st query
        dialog+=[{"role": "user",
                    "content": [{"type": "image"},
                                {"type": "text", "text": query.strip()}]
                }]
    else:
        dialog+=[{"role": "user",
                  "content": [{"type":"text",
                               "text": query}]

        }]
    # add_generation_prompt=True appends <|start_header_id|>assistant<|end_header_id|>
    # to prompt the model to generate answer
    input_text=processor.apply_chat_template(dialog, add_generate_prompt=True)
    
    
    # add_special_tokens=False to avoid adding <|begin_of_text|> token
    input_tensor=processor(image,
                       input_text,
                       add_special_tokens=False,
                       return_tensors="pt").to(model.device)
    output_tokens=model.generate(**input_tensor,max_new_tokens=100)

    # take out input tokens and <|start_header_id|>assistant<|end_header_id|>
    generated_tokens_trimmed = [out_ids[len(in_ids)+3 :] for in_ids, out_ids in zip(input_tensor.input_ids,output_tokens)]
    output_text=processor.decode(generated_tokens_trimmed[0])

    # append model response to dialog history
    dialog.append({"role": "assistant",
                   "content": [{"type":"text", "text": output_text[0].strip()}]})
    
    return output_text,dialog

def load_image(image_input):
    """ Load image from URL or bytes. Raises ValueError if input is invalid. """
    try:
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, str) and image_input.startswith(("http://", "https://")):
            return Image.open(requests.get(image_input, stream=True).raw)
        else:
            raise ValueError("Invalid image input. Provide a URL or a bytes object.")
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")


def main():
    model_ft,processor=load_fine_tune_model()
    
    dialog=[]
    while True:
        # prompt for image input
        image_input=input("Enter your image url, or type 'exit' to quit: ")
        if image_input.lower() == 'exit':
            break

        # get image information
        try:
            image=load_image(image_input)   
            dialog=[]   # reset dialog for a new image
        except Exception as e:
            print(f"Error loading image: {e}")   
            continue # Ask for a valid image again

        while True:
            # user query
            user_query=input("Enter your query, or type 'new image' to change the image, or type 'exit' to quit: ")
            if user_query.lower() == 'exit':
                return # break the inner while loop and also break the outer loop
            elif user_query.lower()=='new image':
                break # break the inner loop and ask for a new image
            else:
                response, dialog = generate(model_ft, image, user_query, processor, dialog)
                print("Assistant:", response)


if __name__=="__main__":
    # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    # url="https://datasets-server.huggingface.co/cached-assets/HuggingFaceM4/the_cauldron/--/847a98a779b1652d65111daf20c972dfcd333605/--/ai2d/train/3/images/image-1d100e9.jpg?Expires=1739855103&Signature=Wn~mDcA2RRpppMw7AYwKWaST~rwjYp~q7FLpJObTZ-M4j5nmfptzNk84qEp87psLcaPI5Q-5mPFeFxdK~1-7DscOa166C5w0pB12w3TFi1d758mNHxkXHcGXhicP~lOor220GvZTOV2d1P-IqN9hM6Z8xfRQlo6~kTUY5pn0EntmgVGZnuC5EFKFKBumJEWOX4ET0n-nT0z~MY5ROyx4huiHFXl287uEB2R7QLUZphYcHKMGRQvCZMIQIam4U8U~MOfug~XlXtCf4F3rzMp490BcGIgdOTMuvhm7MAUCoZ7IHbVBccRUfaSYHOAwgpt-UQy667E9RUpvnwZ-j68cmA__&Key-Pair-Id=K3EI6M078Z3AC3"
    main()
