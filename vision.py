import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor


def save_my_compute_please():
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    image_path="images/captured_image_2.jpg"
    image= Image.open(image_path)

    # url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "analyze this image and tell me what the facial expression of the human in the image is. "}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=30)
    print(processor.decode(output[0]))



save_my_compute_please()