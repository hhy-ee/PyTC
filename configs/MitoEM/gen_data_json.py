import json
old_file = 'configs/MitoEM/im_train.json'
new_file = 'data/MitoEM/MitoEM_H/im_train.json'

old_json = json.load(open(old_file))
img_file = old_json['image']
# for im_xxx.json
img_file = [item.replace('/path/to/MitoEM-R', 'data/MitoEM/MitoEM_H') for item in img_file]
# for gt_xxx.json
# img_file = ['data/MitoEM/MitoEM_H/mito_val/seg{:0>4d}.tif'.format(i) for i in range(400, 500)]
new_json = old_json.copy()
new_json['image'] = img_file

f = open(new_file, 'w')
json.dump(new_json, f)

# import json
# import os
# import sys
# base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(0, base_dir)

# js_path = 'configs/MitoEM/im_train.json'
# my_path = 'data/MitoEM_R/im_train.json'
# with open(js_path, 'r') as fp:
#     data = json.load(fp)

# for i in range(len(data['image'])):
#     x = data['image'][i]
#     x = x.strip().split('/')
#     data['image'][i] = my_path+'/'.join(x[-2:])

# with open(js_path, 'w') as fp:
#     json.dump(data, fp)