import json
old_file = '/home/ps/hhy/MitoEM/PyTC/configs/MitoEM/im_val.json'
new_file = '/home/ps/hhy/MitoEM/PyTC/data/MitoEM/MitoEM_R/im_val.json'

old_json = json.load(open(old_file))
img_file = old_json['image']
# for im_xxx.json
img_file = [item.replace('/path/to/MitoEM-R', 'data/MitoEM/MitoEM_R') for item in img_file]
# for gt_xxx.json
# img_file = ['data/MitoEM/MitoEM_R/mito_train/seg{:0>4d}.tif'.format(i) for i in range(0, 400)]
new_json = old_json.copy()
new_json['image'] = img_file

f = open(new_file, 'w')
json.dump(new_json, f)