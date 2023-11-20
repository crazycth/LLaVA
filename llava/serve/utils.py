from typing import Dict
import torch, os, copy
from PIL import Image
import pydicom
from ..train.train import preprocess_multimodal, preprocess, DataArguments

#DataArguments(data_path='/root/code/LLaVA/medical_image_dataset/test.json', lazy_preprocess=True, is_multimodal=True, image_folder='/root/code/LLaVA', image_aspect_ratio='pad', image_grid_pinpoints=None)

data_args_instance = DataArguments(
    lazy_preprocess=True,
    is_multimodal=True,
    image_aspect_ratio='pad',
    image_grid_pinpoints=None,
    image_folder='./'
)

def load_medical_image(source:Dict, image_folder:str, processor, tokenizer) -> Dict[str, torch.Tensor]:
    print(f"[INFO] source : {source}")
    if 'image' in source:
        images = []
        image_files = source['image']

        for each_image in image_files:
            images_each = []
            for image_file in each_image:
                dcm = pydicom.dcmread(os.path.join(image_folder, image_file))
                image = Image.fromarray(dcm.pixel_array).convert('RGB')

                #do pad
                def expand2square(pil_image, background_color):
                    width, height = pil_image.size
                    if width == height:
                        return pil_image
                    elif width > height:
                        result = Image.new(pil_image.mode, (width,width), background_color)
                        result.paste(pil_image, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_image.mode, (height, height), background_color)
                        result.paste(pil_image, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

                image = image.unsqueeze(0)
                images_each.append(image)

            images_each = torch.cat([image for image in images_each], dim = 0)
            images_each = images_each.unsqueeze(0)
            images.append(images_each)

        images = torch.cat([image for image in images], dim=0)

        data_args_instance.mm_use_im_start_end=True
        source = preprocess_multimodal(
            copy.deepcopy([source["conversations"]]), data_args=data_args_instance
        )

    data_dict = preprocess(source,tokenizer,has_image=True)
    data_dict = dict(
        input_ids=data_dict["input_ids"][0],
        labels=data_dict["labels"][0],
        image=images
    )
    return data_dict