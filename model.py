import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
import numpy as np


def load_vocab(vocab_path):
    vocab_list = []
    with open(vocab_path, 'r') as reader:
        for l in reader.readlines():
            vocab_list.append(l.strip().lower())
    return vocab_list


def load_embedding(embedding_path):
    return np.loadtxt(embedding_path)


class SentenceVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type,
                 hidden_size, word_dropout, embedding_dropout, latent_size,
                 sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length,
                 num_layers, bidirectional,
                 word_vocab, concept_vocab,
                 pretrained_word_embedding_path,
                 pretrained_concept_embedding_path
         ):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if pretrained_word_embedding_path:
            pretrain_word_vocab = load_vocab(pretrained_word_embedding_path + '.vocab')
            pretrain_w2i = dict(zip(pretrain_word_vocab, range(len(pretrain_word_vocab))))
            pretrain_word_embedding = load_embedding(pretrained_word_embedding_path + '.vec')

            wv_matrix = []
            for word_id, word in enumerate(word_vocab):
                if word in pretrain_word_vocab:
                    wv_matrix.append(pretrain_word_embedding[pretrain_w2i[word]])
                else:
                    wv_matrix.append(np.random.uniform(-0.01, 0.01, embedding_size).astype("float32"))

            wv_matrix = np.array(wv_matrix, dtype="float32")
            print('Load pretrained word embedding: embedding.shape=%s, pretrained_embedding.shape=%s' %
                  (str(self.embedding.weight.data.shape), wv_matrix.shape))
            self.embedding.weight.data.copy_(torch.from_numpy(wv_matrix))

        self.concept_embedding = None
        if pretrained_concept_embedding_path:
            self.concept_embedding = nn.Embedding(len(concept_vocab), embedding_size)
            pretrain_concept_vocab = load_vocab(pretrained_concept_embedding_path + '.vocab')
            pretrain_c2i = dict(zip(pretrain_word_vocab, range(len(pretrain_concept_vocab))))
            pretrain_concept_embedding = load_embedding(pretrained_concept_embedding_path + '.vec')

            cv_matrix = []
            for c_id, concept in enumerate(concept_vocab):
                if concept in pretrain_concept_vocab:
                    cv_matrix.append(pretrain_concept_embedding[pretrain_c2i[word]])
                else:
                    cv_matrix.append(np.random.uniform(-0.01, 0.01, embedding_size).astype("float32"))

            cv_matrix = np.array(cv_matrix, dtype="float32")
            print('Load pretrained concept embedding: embedding.shape=%s, pretrained_embedding.shape=%s' %
                  (str(self.concept_embedding.weight.data.shape), cv_matrix.shape))
            self.concept_embedding.weight.data.copy_(torch.from_numpy(cv_matrix))

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        # a linear readout layer for exporting to concept vector and consistency penalty
        self.latent2conceptembedding = nn.Linear(latent_size, embedding_size)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)


    def forward(self, input_sequence, length):

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        variance = self.hidden2logv(hidden)
        std = torch.exp(0.5 * variance)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        # CONCEPT READOUT
        concept_embed = self.latent2conceptembedding(z)

        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            # ensure words of SOS and PAD are not dropped out
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        return logp, mean, variance, z, concept_embed


    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t < self.max_sequence_length and len(running_seqs) > 0):

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                if len(input_sequence.shape) == 0:
                    input_sequence = self.tensor([input_sequence]).long()
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
