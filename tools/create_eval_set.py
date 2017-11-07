
import json
import sys

source = sys.argv[1]
target = sys.argv[2]
num_samples = int(sys.argv[3])


def load_json(json_path):
    with open(json_path, 'r') as outfile:
        return json.load(outfile)


def save_json(data, output_json):
    with open(output_json,  'w') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4, ensure_ascii=False)


def main(true_captions_file, target_file, num_samples):
    eval_captions = {}
    true_captions = load_json(true_captions_file)
    true_anns = {anns['image_id']: anns for anns in true_captions['annotations']}
    counter = 0
    collect_imgs = []
    collect_caps = []
    for img in true_captions['images']:
        counter += 1
        if counter <= num_samples:
            collect_imgs.append(img)
            collect_caps.append(true_anns[img['id']])
        else:
            break

    eval_captions = true_captions
    eval_captions['annotations'] = collect_caps
    eval_captions['images'] = collect_imgs
    eval_captions['type'] = 'captions'

    save_json(eval_captions, target_file)


if __name__ == '__main__':
    main(source, target, num_samples)
