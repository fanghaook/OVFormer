import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default='datasets/LVVIS/train/train_instances_nonovel.json')
    parser.add_argument("--add_freq", action='store_true')

    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cats = data['categories']
    videos = data['videos']

    video_count = {x['id']: set() for x in cats}
    image_count = {x['id']: 0 for x in cats}
    ann_count = {x['id']: 0 for x in cats}

    for x in data['annotations']:
        video_count[x['category_id']].add(x['video_id'])
        ann_count[x['category_id']] += x['length']

    for category_id, video_set in video_count.items():
        for video_id in video_set:
            image_count[category_id] += videos[video_id]['length']

    num_freqs = {x: 0 for x in ['b', 'n']}
    for x in cats:
        x['image_count'] = image_count[x['id']]
        x['instance_count'] = ann_count[x['id']]
        if args.add_freq:
            freq = 'b'
            if x['image_count'] == 0:
                freq = 'n'
            x['frequency'] = freq
            num_freqs[freq] += 1

    if args.add_freq:
        for x in ['b', 'n']:
            print(x, num_freqs[x])
    out = cats # {'categories': cats}
    out_path = 'datasets/metadata/lvvis_train_cat_info.json'
    print('Saving to', out_path)
    json.dump(out, open(out_path, 'w'))
