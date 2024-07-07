import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/LVVIS/train/train_instances_.json')
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    print('all #anns', len(data['annotations']))  # 15967

    novel_categories = [i['id'] for i in data['categories'] if i['partition'] in [2, 3]]
    data['annotations'] = [x for x in data['annotations'] if x['category_id'] not in novel_categories]

    print('nonovel #anns', len(data['annotations']))  # 10884
    out_path = args.ann[:-5] + 'nonovel.json'
    print('Saving to', out_path)
    json.dump(data, open(out_path, 'w'))

