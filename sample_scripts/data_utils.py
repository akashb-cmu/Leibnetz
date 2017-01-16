import numpy as np

def get_one_hot(w_index, vocab):
    vocab_size = len(vocab)
    zero = np.zeros(shape=(vocab_size,), dtype=np.int32)
    zero[w_index] = 1
    return zero

def get_one_hots(line_indices, vocab):
    op_vecs = []
    for w_index in line_indices:
        op_vecs.append(get_one_hot(w_index, vocab))
    op_vecs.append(get_one_hot(len(vocab)-1, vocab))
    return op_vecs

def read_lm_data(file_path, retain_seq_len_batch=False, start_sym="<s>", end_sym="<\s>"):
    vocab = {start_sym:0} # Maps a word to id
    ret_data = {} # Maps a sequence length to the numpy array of sequences of that size
    ret_ops = {}
    max_w_index = 1
    with open(file_path, 'r') as ipf:
        for line in ipf:
            line = line.strip(" \t\r\n")
            words = line.split()
            for start in range(len(words)):
                line_indices = [vocab[start_sym]]
                for word in words[start:]:
                    if vocab.get(word, None) is None:
                        vocab[word] = max_w_index
                        max_w_index += 1
                    line_indices.append(vocab[word])
                if ret_data.get(len(line_indices), None) is None:
                    ret_data[len(line_indices)] = [line_indices]
                else:
                    ret_data[len(line_indices)].append(line_indices)
    vocab[end_sym] = max_w_index
    for key in ret_data.keys():
        for line in ret_data[key]:
            line_len = len(line)
            line.append(vocab[end_sym])
            if ret_ops.get(line_len, None) is None:
                ret_ops[line_len] = [get_one_hots(line[1:], vocab)]
            else:
                ret_ops[line_len].append(get_one_hots(line[1:], vocab))

    inv_vocab = {}
    for key in vocab.keys():
        inv_vocab[vocab[key]] = key

    if retain_seq_len_batch:
        composite_data = {}
        for key in ret_data.keys():
            assert ret_ops.get(key, None) is not None, "Mismatch between ret_data and ret_ops!"
            composite_data[key] = zip(ret_data[key], ret_ops[key])
        return vocab, inv_vocab, composite_data

    else:
        composite_data = []
        for key in ret_data.keys():
            assert ret_ops.get(key, None) is not None, "Mismatch between ret_data and ret_ops!"
            for item in zip(ret_data[key], ret_ops[key]):
                composite_data.append(item)
        return vocab, inv_vocab, composite_data