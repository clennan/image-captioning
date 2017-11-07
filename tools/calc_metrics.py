
import argparse
from src.coco_caption.pycocotools.coco import COCO
from src.coco_caption.pycocoevalcap.eval import COCOEvalCap
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


def load_json(json_path):
    with open(json_path, 'r') as outfile:
        return json.load(outfile)


def save_json(data, output_json):
    with open(output_json,  'w') as outfile:
        json.dump(data, outfile, sort_keys=True, indent=4, ensure_ascii=False)


def convert_result(result_json):
    res = load_json(result_json)

    collect = []
    for item in res:
        new = {}
        new['image_id'] = int(str(item['image_id']).lstrip('0'))
        new['caption'] = ' '.join([i for i in item['caption'].split(' ') if 'sentence_' not in i])
        collect.append(new)

    save_json(collect, result_json)
    print('converted json saved')


def calculate_metrics(generated_captions_file, true_captions_file):
    coco = COCO(true_captions_file)
    cocoRes = coco.loadRes(generated_captions_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        print '%s: %.3f' % (metric, score)

    return coco, cocoEval, cocoRes


def main(generated_captions_file, true_captions_file):
    convert_result(generated_captions_file)
    calculate_metrics(generated_captions_file, true_captions_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--captions-file',
      help='JSON file with generated captions and image IDs',
      required=True
      )

    args = parser.parse_args()
    arguments = args.__dict__

    true_captions_file = 'src/coco_caption/annotations/captions_val2014.json'

    main(arguments['captions_file'], true_captions_file)
