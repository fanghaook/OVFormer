# Split results.json into base categories and novel categories, and zip compress them
# YTVIS19：AP*40 = APb*33 + APn*7
# YTVIS21：AP*40 = APb*34 + APn*6
import json
import zipfile
import os

# novel_id_ytvis19 = [6, 7, 9, 11, 23, 24, 39]
# novel_id_ytvis21 = [11, 14, 15, 20, 30, 39]
novel_id = [6, 7, 9, 11, 23, 24, 39]
novel_list = []
base_list = []

results = json.load(open('output/inference/ytvis_2019_val/results.json', 'r'))
for result in results:
    if result['category_id'] in novel_id:
        novel_list.append(result)
    else:
        base_list.append(result)

# all.zip
file_name = "results.json"
with open(file_name, 'w') as json_file:
    json.dump(results, json_file)
zip_file_name = "all.zip"
with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(file_name)

# novel.zip
with open(file_name, 'w') as json_file:
    json.dump(novel_list, json_file)
zip_file_name = "novel.zip"
with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(file_name)

# base.zip
with open(file_name, 'w') as json_file:
    json.dump(base_list, json_file)
zip_file_name = "base.zip"
with zipfile.ZipFile(zip_file_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(file_name)

# delete results.json
if os.path.exists(file_name):
    os.remove(file_name)