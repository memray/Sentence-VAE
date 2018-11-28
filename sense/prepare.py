# -*- coding: utf-8 -*-
"""
Provide numpy tensor for init tensorflow tensor
"""
import io
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import os

import dataset

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

pretrained_wv_path = '/home/memray/Data/wsd_data/fasttext/fasttext_dim128_epoch50_brute.vec'

BASE_PATH = '/home/memray/Data/wsd_data/'
EMBED_MODEL_NAME = 'fasttext_dim128_epoch50_brute'

PRETRAINED_VEC_PATH = '%s/pretrained_model/%s.vec' % (BASE_PATH, EMBED_MODEL_NAME)
PREV_ABBR_PATH = '%s/mimic/abbr' % (BASE_PATH)
PREV_CUI_PATH = '%s/mimic/cui' % (BASE_PATH)
PREV_CUI_DEF_PATH = '../data/sense/definition/CUI_definition.RRF'

OUTPUT_BASE_PATH = '../data/sense/'
ABBR_VOCAB_PATH = '%s/pretrained_embedding/abbr_vocab' % (OUTPUT_BASE_PATH)
ABBR_EMB_PATH = '%s/pretrained_embedding/abbr.vec' % (OUTPUT_BASE_PATH)

CUI_VOCAB_PATH = '%s/pretrained_embedding/cui_vocab' % (OUTPUT_BASE_PATH)
CUI_EMB_PATH = '%s/pretrained_embedding/cui.vec' % (OUTPUT_BASE_PATH)

WORD_VOCAB_PATH = '%s/pretrained_embedding/word_vocab' % (OUTPUT_BASE_PATH)
WORD_EMB_PATH = '%s/pretrained_embedding/word.vec' % (OUTPUT_BASE_PATH)

# external sense metadata (name and definition), may change to variables later
sense_inventory_path = '../data/sense/umls/sense_inventory_with_testsets.json'
sense_name_path = '../data/sense/umls/CUI_name.RRF'
sense_definition_path = '../data/sense/umls/CUI_definition.RRF'

if not os.path.exists(OUTPUT_BASE_PATH):
    os.makedirs(OUTPUT_BASE_PATH)

# whether it is preprocessed by Byte-Pair Encoding
subvoc = False

def export_vocab_and_pretrained_embedding():
    # load ABBR and CUI vocab (generated during processing MIMIC)
    abbr_vocab = [abbr.strip() for abbr in
                  open(PREV_ABBR_PATH).readlines()]
    abbr2id = dict(zip(abbr_vocab, range(len(abbr_vocab))))
    cui_vocab = [cui.strip() for cui in
                 open(PREV_CUI_PATH).readlines()]
    cui2id = dict(zip(cui_vocab, range(len(cui_vocab))))

    # initialize empty tensors
    abbr_tensor = [[None] for _ in range(len(abbr_vocab))]
    cui_tensor = [[None] for _ in range(len(cui_vocab))]
    word_tensor = []
    word_vocab = []

    # load the pre-traiend word vectors, one word per line
    fin = io.open(PRETRAINED_VEC_PATH, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # 1st line tells number of vectors and dim of each word embedding
    num_embed, embed_dim = map(int, fin.readline().split())
    data = {}
    abbr_count, cui_count, word_count = 0, 0, 0

    # process each line
    for line in fin:
        tokens = line.rstrip().split(' ')

        token = tokens[0]
        vec = [float(v) for v in tokens[1:]]

        # remove the tailing underline if it's a subword
        if subvoc and token[-1] == '_':
            token = token[:-1]

        # an abbr
        if token in abbr2id:
            abbr_tensor[abbr2id[token]] = vec
            abbr_count += 1
        # a CUI
        elif token in cui2id:
            cui_tensor[cui2id[token]] = vec
            cui_count += 1
        # a word
        else:
            word_tensor.append(vec)
            word_vocab.append(token)
            word_count += 1

    print('Load %s embeddings in dim %s' % (num_embed, embed_dim))
    assert cui_count == len(cui_vocab), 'cui_count: %s; id2sense: %s' % (cui_count, len(cui_vocab))
    assert abbr_count == len(abbr_vocab), 'abbr_count: %s; id2abbr: %s' % (abbr_count, len(abbr_vocab))
    assert len(word_vocab) == len(word_tensor) == word_count

    # Append special tokens
    for token in dataset.VOCAB_PRESET_TOKENS:
        word_vocab.append(token)
        word_tensor.append([0.0] * embed_dim)

    # export vectors
    abbr_tensor = np.array(abbr_tensor)
    np.savetxt(ABBR_EMB_PATH, abbr_tensor)
    print('Output abbr tensor with size %s' % abbr_count)

    cui_tensor = np.array(cui_tensor)
    np.savetxt(CUI_EMB_PATH, cui_tensor)
    print('Output cui tensor with size %s' % cui_count)

    word_tensor = np.array(word_tensor)
    np.savetxt(WORD_EMB_PATH, word_tensor)

    # export vocabs
    print('Output vocab tensor with size %s' % word_count)
    if subvoc:
        f = open(WORD_VOCAB_PATH, 'w')
        for wd in word_vocab:
            f.write('\'%s\'\n' % (wd))
        f.close()
    else:
        f = open(WORD_VOCAB_PATH, 'w')
        for wd in word_vocab:
            f.write('%s\n' % (wd))
        f.close()

    f = open(ABBR_VOCAB_PATH, 'w')
    for abbr in abbr_vocab:
        f.write('%s\n' % (abbr))
    f.close()

    f = open(CUI_VOCAB_PATH, 'w')
    for cui in cui_vocab:
        f.write('%s\n' % (cui))
    f.close()

    print('Output vocab with same size')


def load_CUI_values(cui_list, filepath, field):
    cui_set = set(cui_list)
    cui_name_dict = defaultdict(str)
    sense_df = pd.read_csv(filepath, sep='|', header=None, index_col=False)

    for row_index, row in sense_df.iterrows():
        # 1st column is CUI
        cui = row.iloc[0].strip().upper()

        if row_index > 0 and row_index % 10000 == 0:
            print('row #%d' % row_index)
            # break

        # not a CUI we are interested in
        if cui not in cui_set:
            continue
        # only take the 1st name for each sense
        if cui in cui_name_dict:
            continue

        if isinstance(field, str):
            text = str(row[field])
        elif isinstance(field, int):
            text = str(row.iloc[field])

        cui_name_dict[cui] = text

    print('Found %d values for %d CUI' % (len(cui_name_dict), len(cui_set)))
    # assert len(cui_set) == len(cui_name_dict)

    return cui_name_dict


def export_training_definition():
    cui_list        = [cui.strip() for cui in
                 open(CUI_VOCAB_PATH).readlines()]
    cui2id          = dict(zip(cui_list, range(len(cui_list))))

    # cui_list (9k CUIs) is a subset of full_sense_inventory (30k+ CUIs)
    full_sense_inventory = {}
    for l in open(sense_inventory_path, 'r').readlines():
        cui_dict = json.loads(l)
        full_sense_inventory[cui_dict['CUI']] = cui_dict

    cui2name = {}
    for cui in full_sense_inventory:
        if len(full_sense_inventory[cui]['COMMON_NAME']) > 0:
            cui2name[cui] = full_sense_inventory[cui]['COMMON_NAME'][0].strip().lower()
        else:
            cui2name[cui] = full_sense_inventory[cui]['LONGFORM'][0].strip().lower()

    cui2def  = load_CUI_values(list(full_sense_inventory.keys()), filepath=sense_definition_path, field=5) # column 5 is definition

    # export all
    with open(OUTPUT_BASE_PATH + 'cui_name_def.all.txt', 'w') as writer:
        for cui, def_ in cui2def.items():
            name = cui2name[cui]
            writer.write('%s|%s|%s\n' % (cui, name, def_))

    # shuffle usable CUIs and split to train/valid
    cui2def_train_valid = {cui: def_ for cui, def_ in cui2def.items() if cui in cui2id}
    cui2def_train_valid_items = sorted(cui2def_train_valid.items(), key=lambda x:x[0])
    random.Random(2597).shuffle(cui2def_train_valid_items)

    # export train (90% CUIs that have both pretrained context vector and definitions)
    with open(OUTPUT_BASE_PATH + 'cui_name_def.train.txt', 'w') as writer:
        for cui, def_ in cui2def_train_valid_items[: int(len(cui2def_train_valid_items) * 0.9)]:
            name = cui2name[cui]
            writer.write('%s|%s|%s\n' % (cui, name, def_))
        print('Export %d CUIs for training' % int(len(cui2def_train_valid_items) * 0.9))

    # export train (10% CUIs that have both pretrained context vector and definitions)
    with open(OUTPUT_BASE_PATH + 'cui_name_def.valid.txt', 'w') as writer:
        for cui, def_ in cui2def_train_valid_items[int(len(cui2def_train_valid_items) * 0.9): ]:
            name = cui2name[cui]
            writer.write('%s|%s|%s\n' % (cui, name, def_))
        print('Export %d CUIs for validation' % len(cui2def_train_valid_items[int(len(cui2def_train_valid_items) * 0.9): ]))

    # export test (CUIs that have definitions only)
    cui2def_test = {cui: def_ for cui, def_ in cui2def.items() if cui not in cui2id}
    with open(OUTPUT_BASE_PATH + 'cui_name_def.test.txt', 'w') as writer:
        for cui, def_ in cui2def_test.items():
            name = cui2name[cui]
            writer.write('%s|%s|%s\n' % (cui, name, def_))
        print('Export %d CUIs for testing' % (len(cui2def_test)))

if __name__ == '__main__':
    # export_vocab_and_pretrained_embedding()
    export_training_definition()