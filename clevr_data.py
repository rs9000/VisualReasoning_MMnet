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

    def __init__(self, features, question_h5):
        """ Initialize a ClevrDataset object.
        """
        self.features = features

        # Parse Questions
        questions = question_h5['questions']
        image_indices = question_h5['image_idxs']
        programs = None
        if 'programs' in question_h5:
            programs = question_h5['programs']
        answers = question_h5['answers']

        assert len(questions) == len(image_indices) == len(answers), \
            'The questions, image indices, programs, and answers are not all the same size!'

        # questions, image indices, programs, and answers are small enough to load into memory
        self.all_questions = torch.LongTensor(np.asarray(questions))
        self.all_image_idxs = torch.LongTensor(np.asarray(image_indices))
        if programs is not None:
            self.all_programs = torch.LongTensor(np.asarray(programs)) if programs is not None else None
        self.all_answers = torch.LongTensor(np.asarray(answers)) if answers is not None else None

    def __getitem__(self, index):
        question = self.all_questions[index]
        image_idx = self.all_image_idxs[index]
        answer = self.all_answers[index] if self.all_answers is not None else None
        program_seq = self.all_programs[index] if self.all_programs is not None else None
        feats = torch.FloatTensor(np.asarray(self.features['features'][image_idx]))

        return question, image_idx, feats, answer, program_seq

    def __len__(self):
        return self.all_questions.size(0)


class ClevrDataLoader(DataLoader):

    def __init__(self, **kwargs):
        if 'question_h5' not in kwargs:
            raise ValueError('Must give question_h5')
        if 'feature_h5' not in kwargs:
            raise ValueError('Must give feature_h5')

        # Read images features
        feature_h5_path = str(kwargs.pop('feature_h5'))
        print('Reading features from ', feature_h5_path)
        self.features = h5py.File(feature_h5_path, 'r')

        # Read questions
        question_h5_path = str(kwargs.pop('question_h5'))
        print('Reading questions from ', question_h5_path)
        with h5py.File(question_h5_path, 'r') as question_h5:
            self.dataset = ClevrDataset(self.features, question_h5)
        kwargs['collate_fn'] = clevr_collate
        super(ClevrDataLoader, self).__init__(self.dataset, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.features is not None:
            self.features.close()


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
