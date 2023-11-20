# 4张22层3D，每2张对应一份报告
a = {'id': '1.2.840.113619.2.437.3.2831177020.5.1648186977.789', 'image': [['medical_images/SRS00002/IMG00000.DCM', 'medical_images/SRS00002/IMG00001.DCM', 'medical_images/SRS00002/IMG00002.DCM', 'medical_images/SRS00002/IMG00003.DCM', 'medical_images/SRS00002/IMG00004.DCM', 'medical_images/SRS00002/IMG00005.DCM', 'medical_images/SRS00002/IMG00006.DCM', 'medical_images/SRS00002/IMG00007.DCM', 'medical_images/SRS00002/IMG00008.DCM', 'medical_images/SRS00002/IMG00009.DCM', 'medical_images/SRS00002/IMG00010.DCM', 'medical_images/SRS00002/IMG00011.DCM', 'medical_images/SRS00002/IMG00012.DCM', 'medical_images/SRS00002/IMG00013.DCM', 'medical_images/SRS00002/IMG00014.DCM', 'medical_images/SRS00002/IMG00015.DCM', 'medical_images/SRS00002/IMG00016.DCM', 'medical_images/SRS00002/IMG00017.DCM', 'medical_images/SRS00002/IMG00018.DCM', 'medical_images/SRS00002/IMG00019.DCM', 'medical_images/SRS00002/IMG00020.DCM', 'medical_images/SRS00002/IMG00021.DCM'], ['medical_images/SRS00002/IMG00000.DCM', 'medical_images/SRS00002/IMG00001.DCM', 'medical_images/SRS00002/IMG00002.DCM', 'medical_images/SRS00002/IMG00003.DCM', 'medical_images/SRS00002/IMG00004.DCM', 'medical_images/SRS00002/IMG00005.DCM', 'medical_images/SRS00002/IMG00006.DCM', 'medical_images/SRS00002/IMG00007.DCM', 'medical_images/SRS00002/IMG00008.DCM', 'medical_images/SRS00002/IMG00009.DCM', 'medical_images/SRS00002/IMG00010.DCM', 'medical_images/SRS00002/IMG00011.DCM', 'medical_images/SRS00002/IMG00012.DCM', 'medical_images/SRS00002/IMG00013.DCM', 'medical_images/SRS00002/IMG00014.DCM', 'medical_images/SRS00002/IMG00015.DCM', 'medical_images/SRS00002/IMG00016.DCM', 'medical_images/SRS00002/IMG00017.DCM', 'medical_images/SRS00002/IMG00018.DCM', 'medical_images/SRS00002/IMG00019.DCM', 'medical_images/SRS00002/IMG00020.DCM', 'medical_images/SRS00002/IMG00021.DCM'], ['medical_images/SRS00002/IMG00000.DCM', 'medical_images/SRS00002/IMG00001.DCM', 'medical_images/SRS00002/IMG00002.DCM', 'medical_images/SRS00002/IMG00003.DCM', 'medical_images/SRS00002/IMG00004.DCM', 'medical_images/SRS00002/IMG00005.DCM', 'medical_images/SRS00002/IMG00006.DCM', 'medical_images/SRS00002/IMG00007.DCM', 'medical_images/SRS00002/IMG00008.DCM', 'medical_images/SRS00002/IMG00009.DCM', 'medical_images/SRS00002/IMG00010.DCM', 'medical_images/SRS00002/IMG00011.DCM', 'medical_images/SRS00002/IMG00012.DCM', 'medical_images/SRS00002/IMG00013.DCM', 'medical_images/SRS00002/IMG00014.DCM', 'medical_images/SRS00002/IMG00015.DCM', 'medical_images/SRS00002/IMG00016.DCM', 'medical_images/SRS00002/IMG00017.DCM', 'medical_images/SRS00002/IMG00018.DCM', 'medical_images/SRS00002/IMG00019.DCM', 'medical_images/SRS00002/IMG00020.DCM', 'medical_images/SRS00002/IMG00021.DCM'], ['medical_images/SRS00002/IMG00000.DCM', 'medical_images/SRS00002/IMG00001.DCM', 'medical_images/SRS00002/IMG00002.DCM', 'medical_images/SRS00002/IMG00003.DCM', 'medical_images/SRS00002/IMG00004.DCM', 'medical_images/SRS00002/IMG00005.DCM', 'medical_images/SRS00002/IMG00006.DCM', 'medical_images/SRS00002/IMG00007.DCM', 'medical_images/SRS00002/IMG00008.DCM', 'medical_images/SRS00002/IMG00009.DCM', 'medical_images/SRS00002/IMG00010.DCM', 'medical_images/SRS00002/IMG00011.DCM', 'medical_images/SRS00002/IMG00012.DCM', 'medical_images/SRS00002/IMG00013.DCM', 'medical_images/SRS00002/IMG00014.DCM', 'medical_images/SRS00002/IMG00015.DCM', 'medical_images/SRS00002/IMG00016.DCM', 'medical_images/SRS00002/IMG00017.DCM', 'medical_images/SRS00002/IMG00018.DCM', 'medical_images/SRS00002/IMG00019.DCM', 'medical_images/SRS00002/IMG00020.DCM', 'medical_images/SRS00002/IMG00021.DCM']], 'conversations': [{'from': 'human', 'value': '<image><image>This photo is 2023.6.1. Render a clear and concise summary of the photo.\n'}, {'from': 'gpt', 'value': 'select luxury furniture 3 - inch gel memory foam mattress topper'}]}


# 4张2D，每2张对应一份报告
b = {
    'id': '1.2.840.113619.2.437.3.2831177020.5.1648186977.789', 
    'conversations': [
    {'from': 'human', 'value': 'This photo is 2023.6.1. Render a clear and concise summary of the photo.\n'}, 
    {'from': 'gpt', 'value': 'select luxury furniture 3 - inch gel memory foam mattress topper'}, 
    {'from': 'human', 'value': 'This photo is 2023.6.7. Render a clear and concise summary of the photo.\n'}, 
    {'from': 'gpt', 'value': 'Compared to 6.1. select luxury furniture 3 - inch gel memory foam mattress topper'}
    ]
    }

train_images = []

for i in range(1000):
    train_images.append(a)
    train_images.append(b)

import json
save_path = 'medical_image_dataset/test_text.json'

with open(save_path, 'w') as w:
    json.dump(train_images, w, indent=3)