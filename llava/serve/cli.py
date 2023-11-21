import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model, load_medical_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
from llava.serve.utils import load_medical_image

data_point = {
      "id": "1.2.840.113619.2.437.3.2831177020.5.1648186977.789",
      "image": [
         [
            "medical_images/SRS00002/IMG00000.DCM",
            "medical_images/SRS00002/IMG00001.DCM",
            "medical_images/SRS00002/IMG00002.DCM",
            "medical_images/SRS00002/IMG00003.DCM",
            "medical_images/SRS00002/IMG00004.DCM",
            "medical_images/SRS00002/IMG00005.DCM",
            "medical_images/SRS00002/IMG00006.DCM",
            "medical_images/SRS00002/IMG00007.DCM",
            "medical_images/SRS00002/IMG00008.DCM",
            "medical_images/SRS00002/IMG00009.DCM",
            "medical_images/SRS00002/IMG00010.DCM",
            "medical_images/SRS00002/IMG00011.DCM",
            "medical_images/SRS00002/IMG00012.DCM",
            "medical_images/SRS00002/IMG00013.DCM",
            "medical_images/SRS00002/IMG00014.DCM",
            "medical_images/SRS00002/IMG00015.DCM",
            "medical_images/SRS00002/IMG00016.DCM",
            "medical_images/SRS00002/IMG00017.DCM",
            "medical_images/SRS00002/IMG00018.DCM",
            "medical_images/SRS00002/IMG00019.DCM",
            "medical_images/SRS00002/IMG00020.DCM",
            "medical_images/SRS00002/IMG00021.DCM"
         ],
         [
            "medical_images/SRS00002/IMG00000.DCM",
            "medical_images/SRS00002/IMG00001.DCM",
            "medical_images/SRS00002/IMG00002.DCM",
            "medical_images/SRS00002/IMG00003.DCM",
            "medical_images/SRS00002/IMG00004.DCM",
            "medical_images/SRS00002/IMG00005.DCM",
            "medical_images/SRS00002/IMG00006.DCM",
            "medical_images/SRS00002/IMG00007.DCM",
            "medical_images/SRS00002/IMG00008.DCM",
            "medical_images/SRS00002/IMG00009.DCM",
            "medical_images/SRS00002/IMG00010.DCM",
            "medical_images/SRS00002/IMG00011.DCM",
            "medical_images/SRS00002/IMG00012.DCM",
            "medical_images/SRS00002/IMG00013.DCM",
            "medical_images/SRS00002/IMG00014.DCM",
            "medical_images/SRS00002/IMG00015.DCM",
            "medical_images/SRS00002/IMG00016.DCM",
            "medical_images/SRS00002/IMG00017.DCM",
            "medical_images/SRS00002/IMG00018.DCM",
            "medical_images/SRS00002/IMG00019.DCM",
            "medical_images/SRS00002/IMG00020.DCM",
            "medical_images/SRS00002/IMG00021.DCM"
         ],
         [
            "medical_images/SRS00002/IMG00000.DCM",
            "medical_images/SRS00002/IMG00001.DCM",
            "medical_images/SRS00002/IMG00002.DCM",
            "medical_images/SRS00002/IMG00003.DCM",
            "medical_images/SRS00002/IMG00004.DCM",
            "medical_images/SRS00002/IMG00005.DCM",
            "medical_images/SRS00002/IMG00006.DCM",
            "medical_images/SRS00002/IMG00007.DCM",
            "medical_images/SRS00002/IMG00008.DCM",
            "medical_images/SRS00002/IMG00009.DCM",
            "medical_images/SRS00002/IMG00010.DCM",
            "medical_images/SRS00002/IMG00011.DCM",
            "medical_images/SRS00002/IMG00012.DCM",
            "medical_images/SRS00002/IMG00013.DCM",
            "medical_images/SRS00002/IMG00014.DCM",
            "medical_images/SRS00002/IMG00015.DCM",
            "medical_images/SRS00002/IMG00016.DCM",
            "medical_images/SRS00002/IMG00017.DCM",
            "medical_images/SRS00002/IMG00018.DCM",
            "medical_images/SRS00002/IMG00019.DCM",
            "medical_images/SRS00002/IMG00020.DCM",
            "medical_images/SRS00002/IMG00021.DCM"
         ],
         [
            "medical_images/SRS00002/IMG00000.DCM",
            "medical_images/SRS00002/IMG00001.DCM",
            "medical_images/SRS00002/IMG00002.DCM",
            "medical_images/SRS00002/IMG00003.DCM",
            "medical_images/SRS00002/IMG00004.DCM",
            "medical_images/SRS00002/IMG00005.DCM",
            "medical_images/SRS00002/IMG00006.DCM",
            "medical_images/SRS00002/IMG00007.DCM",
            "medical_images/SRS00002/IMG00008.DCM",
            "medical_images/SRS00002/IMG00009.DCM",
            "medical_images/SRS00002/IMG00010.DCM",
            "medical_images/SRS00002/IMG00011.DCM",
            "medical_images/SRS00002/IMG00012.DCM",
            "medical_images/SRS00002/IMG00013.DCM",
            "medical_images/SRS00002/IMG00014.DCM",
            "medical_images/SRS00002/IMG00015.DCM",
            "medical_images/SRS00002/IMG00016.DCM",
            "medical_images/SRS00002/IMG00017.DCM",
            "medical_images/SRS00002/IMG00018.DCM",
            "medical_images/SRS00002/IMG00019.DCM",
            "medical_images/SRS00002/IMG00020.DCM",
            "medical_images/SRS00002/IMG00021.DCM"
         ]
      ],
      "conversations": [
         {
            "from": "human",
            "value": "<image><image>This photo is 2023.6.1. Render a clear and concise summary of the photo.\n"
         },
         {
            "from": "gpt",
            "value": "select luxury furniture 3 - inch gel memory foam mattress topper"
         },
         {
            "from": "human",
            "value": "<image><image>This photo is 2023.6.7. Render a clear and concise summary of the photo.\n"
         },
         {
            "from": "gpt",
            "value": "Compared to 6.1. select luxury furniture 3 - inch gel memory foam mattress topper"
         }
      ]
}


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def debug_test(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    print(f"[INFO] model_name: {model_name}, model_path: {args.model_path}, model_base:{args.model_base}")
    # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    tokenizer, model, image_processor, context_len = load_medical_model(args.model_path, model_name, device=args.device)

    print(f"[INFO] start load medical _image")
    print(f"[INFO] image_processor: {image_processor}")
    medical_image_tensor = load_medical_image(
        source=data_point,
        image_folder="/root/code/LLaVA",
        processor=image_processor,
        tokenizer=tokenizer
    )

    print(f"[DEBUG] medical_image_tensor: {medical_image_tensor['image'].shape}")


    print(model)

    for name,para in model.named_parameters():
        print(name, para)


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_medical_model(args.model_path, model_name, device=args.device)

    conv_mode = "llava_llama_2"
    args.conv_mode = conv_mode
    # if 'llama-2' in model_name.lower():
    #     conv_mode = "llava_llama_2"
    # elif "v1" in model_name.lower():
    #     conv_mode = "llava_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    # else:
    #     conv_mode = "llava_v0"

    # if args.conv_mode is not None and conv_mode != args.conv_mode:
    #     print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    # else:
    #     args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image_tensor = load_medical_image(
        source=data_point,
        image_folder="./",
        processor=image_processor,
        tokenizer=tokenizer
    )

    image = image_tensor['image'].to(model.device, dtype=torch.float16)
    image_num = image.shape[0]
    image = torch.unsqueeze(image, dim=0)
    print(f"[INFO] image_tensor shape: {image.shape}")

    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            inp_total = ""
            for _ in range(image_num):
                inp_total += DEFAULT_IMAGE_TOKEN
            inp_total += inp
            conv.append_message(conv.roles[0],inp_total)
            # first message
            # if model.config.mm_use_im_start_end:
            #     inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            # else:
            #     inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            # conv.append_message(conv.roles[0], inp)
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        print(f"[DEBUG] prompt: {prompt}")

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # print(f"[DEBUG] image_tensor now: {image_tensor}\n shape: {image_tensor.shape}")    #[1,3,336,336]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    # debug_test(args)
    main(args)
