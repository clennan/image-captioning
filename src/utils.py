
import matplotlib
matplotlib.use('TkAgg')
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.transform


def load_json(json_path):
    with open(json_path, 'r') as outfile:
        return json.load(outfile)


def convert_index_sequence_to_sentence(
    list_of_sequences, dict_idx_to_word, end_of_sentence_token=None,
    ):
    list_of_sentences = []
    for s, sequence in enumerate(list_of_sequences):
        temp = [dict_idx_to_word[i] for i in sequence]
        if end_of_sentence_token:
            try:
                eos_index = temp.index(end_of_sentence_token)
                temp = temp[:eos_index+1]
            except ValueError:
                temp = temp

        list_of_sentences.append(' '.join(temp))

    return list_of_sentences


def heatmap(x):
    x = np.squeeze(x)
    x = np.sum(x, axis=2)

    x_pos = np.maximum(0, x)
    x_neg = np.minimum(0, x)

    x_pos = x_pos / x_pos.max()
    x_pos = x_pos[..., np.newaxis]
    r = 0.9 - np.clip(x_pos-0.3,0,0.7)/0.7*0.5
    g = 0.9 - np.clip(x_pos-0.0,0,0.3)/0.3*0.5 - np.clip(x_pos-0.3,0,0.7)/0.7*0.4
    b = 0.9 - np.clip(x_pos-0.0,0,0.3)/0.3*0.5 - np.clip(x_pos-0.3,0,0.7)/0.7*0.4

    x_neg = x_neg * -1.0
    x_neg = x_neg / (x_neg.max() + 1e-9)
    x_neg = x_neg[..., np.newaxis]
    r2 = 0.9 - np.clip(x_neg-0.0,0,0.3)/0.3*0.5 - np.clip(x_neg-0.3,0,0.7)/0.7*0.4
    g2 = 0.9 - np.clip(x_neg-0.0,0,0.3)/0.3*0.5 - np.clip(x_neg-0.3,0,0.7)/0.7*0.4
    b2 = 0.9 - np.clip(x_neg-0.3,0,0.7)/0.7*0.5

    return np.concatenate([r, g, b], axis=-1) + np.concatenate([r2, g2, b2], axis=-1)


def vgg_addmean(image_batch):
    vgg_mean = [103.939, 116.779, 123.68]
    img_trans = []
    for img in image_batch:
        blue, green, red = np.split(img, 3, 2)
        img_trans.append(np.concatenate([blue + vgg_mean[0], green + vgg_mean[1], red + vgg_mean[2]], 2))
    return img_trans


def bgr_to_rgb(image_batch):
    img_trans = []
    for img in image_batch:
        blue, green, red = np.split(img, 3, 2)
        img_trans.append(np.concatenate([red, green, blue], 2))
    return img_trans


def plot_columns_dtdc(
    image,
    gen_caption,
    alphas,
    relevances_alpha,
    save_name,
    approach,
    ):

    if not os.path.isdir('results'):
        os.mkdir('results')

    num_words = len(gen_caption.split(' '))
    fig = plt.figure(figsize=(4, num_words+6))

    plt.subplot2grid(((num_words+1)*3+1, 3*2), (0, 0), colspan=3, rowspan=3)
    plt.imshow(image/255.0)
    plt.axis('off')

    # Plot images with attention weights
    for w, word in enumerate(gen_caption.split(' ')):
        # plot image
        plt.subplot2grid(((num_words+1)*3+1, 3*2), (w*3+4, 0),
                         colspan=3, rowspan=3)
        plt.text(0, 0, '%s' % (word), color='black',
                 backgroundcolor='white', fontsize=6)
        plt.imshow(image/255.0)

        # plot alphas
        alpha_current = alphas[w+1].reshape(14, 14)
        alpha_img = skimage.transform.pyramid_expand(
            alpha_current, upscale=16, sigma=20,
            )
        plt.imshow(alpha_img, alpha=0.95, cmap='gray')
        plt.axis('off')
        if w == 0:
            plt.title('$\\alpha_{:}$', fontsize=8)

        # plot DTD-C
        plt.subplot2grid(((num_words+1)*3+1, 3*2), (w*3+4, 3),
                         colspan=3, rowspan=3)
        rt_current = heatmap(relevances_alpha[w+1])
        plt.imshow(rt_current)
        plt.axis('off')
        if w == 0:
            plt.title(approach, fontsize=8)

    fig.savefig(os.path.join('results', save_name+'.pdf'),
                bbox_inches='tight')
    plt.close()

    print('plot saved under %s'
          % (os.path.join('results', save_name+'.pdf')))
