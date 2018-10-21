import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torch
import h5py
import warnings
import json


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])

    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2

    return vocab


class ClevrDataset(Dataset):
    """ Holds a handle to the CLEVR dataset.

    Extended Summary
    ----------------
    A :class:`ClevrDataset` holds a handle to the CLEVR dataset. It loads a specified subset of the
    questions, their image indices and extracted image features, the answer (if available), and
    optionally the images themselves. This is best used in conjunction with a
    :class:`ClevrDataLoaderNumpy` of a :class:`ClevrDataLoaderH5`, which handle loading the data.
    """

    def __init__(self, **kwargs):
        """ Initialize a ClevrDataset object.
        """

        if 'question_h5' not in kwargs:
            raise ValueError('Must give question_h5')
        if 'feature_h5' not in kwargs:
            raise ValueError('Must give feature_h5')

        # Read images features
        feature_h5_path = str(kwargs.pop('feature_h5'))
        print('Reading features from ', feature_h5_path)
        self.features = h5py.File(feature_h5_path, 'r')['features']

        self.images = None
        if 'image_h5' in kwargs:
            image_h5_path = str(kwargs.pop('image_h5'))
            print('Reading images from ', image_h5_path)
            self.images = h5py.File(image_h5_path, 'r')['images']

        indices = None
        if 'indices' in kwargs:
            indices = kwargs.pop('indices')

        if 'shuffle' not in kwargs:
            # be nice, and make sure the user knows they aren't shuffling
            warnings.warn('\n\n\tYou have not provided a \'shuffle\' argument to the data loader.\n'
                          '\tBe aware that the default behavior is to NOT shuffle the data.\n')

        # Read Questions
        question_h5_path = str(kwargs.pop('question_h5'))
        with h5py.File(question_h5_path) as question_h5:
            questions = question_h5['questions']
            image_indices = question_h5['image_idxs']
            programs = question_h5['programs']
            answers = question_h5['answers']

            assert len(questions) == len(image_indices) == len(programs) == len(answers), \
                'The questions, image indices, programs, and answers are not all the same size!'

            # questions, image indices, programs, and answers are small enough to load into memory
            self.all_questions = torch.LongTensor(np.asarray(questions))
            self.all_image_idxs = torch.LongTensor(np.asarray(image_indices))
            self.all_programs = torch.LongTensor(np.asarray(programs))
            self.all_answers = torch.LongTensor(np.asarray(answers)) if answers is not None else None

        if indices is not None:
            indices = torch.LongTensor(np.asarray(indices))
            self.all_questions = self.all_questions[indices]
            self.all_image_idxs = self.all_image_idxs[indices]
            self.all_programs = self.all_programs[indices]
            self.all_answers = self.all_answers[indices]

    def __getitem__(self, index):
        question = self.all_questions[index]
        image_idx = self.all_image_idxs[index]
        answer = self.all_answers[index] if self.all_answers is not None else None
        program_seq = self.all_programs[index]
        image = None
        if self.images is not None:
            image = torch.FloatTensor(np.asarray(self.images[image_idx]))
        feats = torch.FloatTensor(np.asarray(self.features[image_idx]))

        return question, image, feats, answer, program_seq

    def __len__(self):
        return len(self.all_questions)


def clevr_collate(batch):
    """ Collate a batch of data."""
    transposed = list(zip(*batch))
    question_batch = default_collate(transposed[0])
    image_batch = transposed[1]
    if any(img is not None for img in image_batch):
        image_batch = default_collate(image_batch)
    feat_batch = transposed[2]
    if any(f is not None for f in feat_batch):
        feat_batch = default_collate(feat_batch)
    answer_batch = default_collate(transposed[3]) if transposed[3][0] is not None else None
    program_seq_batch = transposed[4]
    if transposed[4][0] is not None:
        program_seq_batch = default_collate(transposed[4])
    return [question_batch, image_batch, feat_batch, answer_batch, program_seq_batch]


def get_loader(**kwargs):

    dataset = ClevrDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=kwargs.pop('batch_size'),
                            shuffle=True, num_workers=kwargs.pop('num_workers'), collate_fn=clevr_collate)
    return dataloader
