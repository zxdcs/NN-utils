import numpy


class Dataset:
    """
    Dataset structure
    """

    def __init__(self, data, shuffle=False, pad_keys=None):
        """
        :param data: A dict of different elements in data
        :param shuffle: Whether to shuffle the dataset or not
        :param pad_keys: The keys of elements that need padding
        """
        self._data = {key: numpy.array(val) for key, val in data.items()}
        self._epochs_completed = 0
        self._index = numpy.arange(len(next(iter(self._data.values()))))
        self._num_examples = len(self._index)
        self._shuffle = shuffle
        self._batches = []
        self._pad_keys = pad_keys if pad_keys else []
        self._seq_lens = []

    def epochs_completed(self):
        return self._epochs_completed

    def get_batches(self, batch_size, keys):
        """
        Get all the batches of the given keys in dataset.
        If shuffle is False, the batched is calculated only once.
        :param batch_size: Batch size
        :param keys: The name of elements to obtain
        :return: A dict of batches indexed by name, and the autural length of the last dimension in batch
        """
        if self._shuffle:
            numpy.random.shuffle(self._index)
            self._build_batches(batch_size, keys)
        else:
            if not self._batches:
                self._build_batches(batch_size, keys)
        self._epochs_completed += 1
        return self._batches, self._seq_lens

    def _build_batches(self, batch_size, keys):
        self._batches = []
        for start in range(0, self._num_examples, batch_size):
            batch = {}
            batch_seq_lens = {}
            for key in keys:
                batch_val = self._data[key][self._index[start:start + batch_size]]
                if key in self._pad_keys:
                    batch_val, seq_lens_val = pad(batch_val)
                    batch_seq_lens[key] = seq_lens_val
                batch[key] = batch_val
            self._batches.append(batch)
            self._seq_lens.append(batch_seq_lens)


def pad(var_arr, max_len=None, fix_len=False):
    """
    Padding the last dimension of a 2D list
    :param var_arr: A 2D list
    :param max_len: The max padding length, list with longer length is trimmed.
    :param fix_len: If max_len is set, it indicate whether the padding length is fixed to max_len.
    :return: A padded 2D numpy array, and the actural length of the last dimention
    """
    if max_len is None:
        seq_len = [len(x) for x in var_arr]
        max_len = max(seq_len)
    else:
        assert max_len > 0
        seq_len = [min(len(x), max_len) for x in var_arr]
        if not fix_len:
            max_len = max(seq_len)
    fixed_var = numpy.zeros((len(var_arr), max_len), dtype='int32')
    for idx, x in enumerate(var_arr):
        fixed_var[idx][0:seq_len[idx]] = x[0:seq_len[idx]]
    return fixed_var, seq_len


def mask(lengths):
    """
    Generate masks that element is 1 if its position < length
    :param lengths: Containing length in each dimenstion.
    :return: A masked numpy array
    """
    lengths = numpy.array(lengths, dtype=numpy.int32)
    lengths = lengths.reshape(lengths.shape[0], -1)
    max_lens = lengths.max(axis=0)
    batch_size = lengths.shape[0]
    mask_size = [batch_size] + max_lens.tolist()
    masks = numpy.zeros(mask_size, dtype=numpy.float32)
    num_dim = len(max_lens)
    assert num_dim < 3
    for sample, length in zip(masks, lengths):
        if num_dim == 1:
            sample[:length[0]] = 1
        elif num_dim == 2:
            sample[:length[0], :length[1]] = 1
    return masks
