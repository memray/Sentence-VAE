"""
Provide numpy tensor for init tensorflow tensor
"""
import io
import numpy as np
from util import constant


PATH_PRETRAINED_VEC = '/exp_data/wsd_data/pretrained_model/fasttext_subtok_dim128_epoch100.vec'
PATH_ABBR = '/exp_data/wsd_data/mimic/abbr'
PATH_CUI = '/exp_data/wsd_data/mimic/cui'

EMB_LABEL = 'fasttext_subtok_dim128_epoch100'
OUTPUT_PATH_VOCAB = '/exp_data/wsd_data/pretrained_model/%s/vocab' % EMB_LABEL
OUTPUT_PATH_ABBR_EMB = '/exp_data/wsd_data/pretrained_model/%s/abbr.vec' % EMB_LABEL
OUTPUT_PATH_CUI_EMB = '/exp_data/wsd_data/pretrained_model/%s/cui.vec' % EMB_LABEL
OUTPUT_PATH_VOCAB_EMB = '/exp_data/wsd_data/pretrained_model/%s/vocab.vec' % EMB_LABEL
subvoc = True

if __name__ == '__main__':
    id2abbr = [abbr.strip() for abbr in
               open(PATH_ABBR).readlines()]
    abbr2id = dict(zip(id2abbr, range(len(id2abbr))))
    id2sense = [cui.strip() for cui in
                     open(PATH_CUI).readlines()]
    sense2id = dict(zip(id2sense, range(len(id2sense))))

    abbr_tensor, cui_tensor, vocab_tensor, vocab = [[None] for _ in range(len(id2abbr))], [[None] for _ in range(len(id2sense))], [], []
    fin = io.open(PATH_PRETRAINED_VEC, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    cnt_abbr, cnt_cui, cnt_vocab = 0, 0, 0
    for line in fin:
        tokens = line.rstrip().split(' ')

        token = tokens[0]
        vec = [float(v) for v in tokens[1:]]

        ntoken = None
        if subvoc and token[-1] == '_':
            ntoken = token[:-1]

        if ntoken in abbr2id:
            abbr_tensor[abbr2id[ntoken]] = vec
            cnt_abbr += 1
        if ntoken in sense2id:
            cui_tensor[sense2id[ntoken]] = vec
            cnt_cui += 1
        else:
            vocab_tensor.append(vec)
            vocab.append(token)
            cnt_vocab += 1

    print('Load Vec %s with dim %s' % (n, d))
    assert cnt_cui == len(id2sense), 'cnt_cui:%s; id2sense:%s' % (cnt_cui, len(id2sense))
    assert cnt_abbr == len(id2abbr), 'cnt_abbr:%s; id2abbr:%s' % (cnt_abbr, len(id2abbr))
    assert len(vocab) == len(vocab_tensor) == cnt_vocab

    # Append pad
    vocab.append(constant.PAD)
    vocab_tensor.append([0.0] * d)

    abbr_tensor = np.array(abbr_tensor)
    np.savetxt(OUTPUT_PATH_ABBR_EMB, abbr_tensor)
    print('Output abbr tensor with size %s.' % cnt_abbr)

    cui_tensor = np.array(cui_tensor)
    np.savetxt(OUTPUT_PATH_CUI_EMB, cui_tensor)
    print('Output cui tensor with size %s' % cnt_cui)

    vocab_tensor = np.array(vocab_tensor)
    np.savetxt(OUTPUT_PATH_VOCAB_EMB, vocab_tensor)

    print('Output vocab tensor with size %s' % cnt_vocab)

    if subvoc:
        f = open(OUTPUT_PATH_VOCAB, 'w')
        for wd in vocab:
            f.write('\'%s\'\n' % (wd))
        f.close()
    else:
        f = open(OUTPUT_PATH_VOCAB, 'w')
        for wd in vocab:
            f.write('%s\t%s\n' % (wd, str(1000)))
        f.close()
    print('Output vocab with same size')
