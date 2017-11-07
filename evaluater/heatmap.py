
import os
import numpy as np
import src.caption_generator as cg
import argparse
from src.utils import convert_index_sequence_to_sentence
from src.utils import vgg_addmean
from src.utils import bgr_to_rgb
from src.utils import plot_columns_dtdc


def run_pipeline(arguments):
    if arguments.get('alpha') is None:
        alpha = 2
    else:
        alpha = int(arguments.get('alpha'))

    approach = arguments['approach']

    # load dictionary with top num_words to index
    word_idx_mapping = np.load('data/dict_top_words_to_index.npy').item()

    # create dictionary index to word
    idx_word_mapping = {v: k for k, v in word_idx_mapping.iteritems()}

    # initialize cap gen class
    cap_gen = cg.CaptionGenerator(
        path_to_pretrained_model=arguments['pretrained_model_folder'],
        )

    # propagte validation image through image captioning system
    outputs, cp, weights, val_img, am = cap_gen.get_outputs(
        arguments['image_file'],
        )

    # convert indices to captions
    caption_indices = zip(*outputs['caption_indices'])

    collect = []
    for c, caption in enumerate(caption_indices):
        collect.append(convert_index_sequence_to_sentence(
            [caption], idx_word_mapping, 'sentence_end'
            ))

    # prepare alphas
    alphas = np.array(outputs['a_'])

    # run dtd
    relevances_c = cap_gen.run_dtd(
        outputs,
        cp,
        weights,
        val_img,
        approach,
        alpha=alpha,
        )

    val_img = vgg_addmean(val_img)
    val_img = bgr_to_rgb(val_img)

    if approach == 'dtdc':
        save_name = os.path.basename(arguments['image_file']).split('.')[0]+'_'+approach+'_alpha'+str(alpha)
    else:
        save_name = os.path.basename(arguments['image_file']).split('.')[0]+'_'+approach

    # plot
    title = (approach[:3]+'-'+approach[-1]).upper()
    plot_columns_dtdc(
        val_img[0],
        ' '.join(collect[0][0].split(' ')[1:-1]),
        alphas,
        relevances_c,
        save_name,
        title,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--image-file',
      help='',
      required=True
      )

    parser.add_argument(
      '--pretrained-model-folder',
      help='folder where pretrained model is saved',
      required=True
      )

    parser.add_argument(
      '--approach',
      help='DTD approach (dtda, dtdb, or dtdc)',
      required=True
      )

    parser.add_argument(
      '--alpha',
      help='alpha that will be used in DTD-C (NB: beta = alpha - 1)',
      required=False
      )

    args = parser.parse_args()
    arguments = args.__dict__

    run_pipeline(arguments)
