"""
This module contains general use functions, global constants and
dependencies.
"""
# coding: utf-8
import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Tuple, Iterator
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
import matplotlib 
import matplotlib.pyplot as plt
import itertools

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
KINGDOM_ENCODING = {
    "Animal": 0,
    "Plant": 1,
    "Fungi": 2,
    "Other": 3,  
    "Kingdom_0": 0,
    "Kingdom_1": 1,
    "Kingdom_2": 2,
    "Kingdom_3": 3,   
}

KINGDOM_KINGDOM = {
    "Animal": "Animal",
    "Plant": "Plant",
    "Fungi": "Fungi",
    "Other": "Other",  
    "Kingdom_0": "Animal",
    "Kingdom_1": "Plant",
    "Kingdom_2": "Fungi",
    "Kingdom_3": "Other",   
}

TERMINAL_SYMBOL = 'Z'
PADDING_SYMBOL = 'X'
AMINO_ENCODING = {
    TERMINAL_SYMBOL:0,
    'A':1,  # Alanine
    'C':2,  # Cystine
    'D':3,  # Aspartic acid
    'E':4,  # Glutamic acid
    'F':5,  # Phenylalanine
    'G':6,  # Glycine
    'H':7,  # Histidine
    'I':8,  # Isoleucine 
    'K':9,  # Lysine
    'L':10, # Leucine
    'M':11, # Methionine
    'N':12, # Asparagine
    'P':13, # Proline
    'Q':14, # Glutamine
    'R':15, # Arginine
    'S':16, # Serine
    'T':17, # Threonine
    'V':18, # Valine
    'W':19, # Tryptophan
    'Y':20, # Tyrosine
    PADDING_SYMBOL:21
}

AMINO_REVERSE_ENCODING = {
    0:TERMINAL_SYMBOL,
    1:'A',
    2:'C',
    3:'D',
    4:'E',
    5:'F',
    6:'G',
    7:'H',
    8:'I',
    9:'K',
    0:'L',
    11:'M',
    12:'N',
    13:'P',
    14:'Q',
    15:'R',
    16:'S',
    17:'T',
    18:'V',
    19:'W',
    20:'Y',
    21:PADDING_SYMBOL
}

## This is taken from this example
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm_nonrmalized = cm
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "%.2f (%d)" % (cm[i, j],cm_nonrmalized[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()

def get_variable(tensor_x: torch.Tensor) -> torch.Tensor:
    """ Converts tensors to cuda, if available. """
    if USE_CUDA:
        return tensor_x.cuda()
    return tensor_x


def get_numpy(tensor_x: torch.Tensor) -> np.ndarray:
    """ Returns the input tensor as a numpy array. """
    if USE_CUDA:
        return tensor_x.cpu().data.numpy()
    return tensor_x.data.numpy()


def make_input_101(amino_str: str) -> str:
    """ Appends the terminal symbol ' Z' to the input string
        and extends it with the padding symbol ' X' if needed. """
    longer_str = amino_str + " " + TERMINAL_SYMBOL
    while len(longer_str) < 201:
        longer_str += " X"
    return longer_str


def make_output_101_class_index(out_str: str) -> int:
    """ out_str:    String of space seperated 0 and 1.
        Returns the Zero based position (index) of the first 1, 
        (spaces ignored). If no 1 encountered it returns the last
        index + 1."""
    for i, x in enumerate(out_str.split()):
        intx = int(x)
        if intx == 1:
            return i
    return i+1


def encode_aminos(amino_str: str) ->Iterator[int]:
    """ amino_str:   Sequence of symbols.
        If a symbol is in AMINO_ENCODING then the matching encoding
        is returned, otherwise the PADDING_SYMBOL's encoding is
        returned. """ 
    spl = amino_str.split()
    if len(spl) == 1:
        spl = spl[0]
    for x in spl:
        if x in AMINO_ENCODING:
            yield AMINO_ENCODING[x]
        else:
            yield AMINO_ENCODING[PADDING_SYMBOL]


def extend_by_sampling(amino_str: str, desired_length: int) -> str:
    while len(amino_str) < desired_length:
        amino_str += np.random.choice(amino_str)
    return amino_str


def fetch_data(fetching: str='train_val', file_type: str='jjaa-style', kingdom_override: str=None) -> List[dict]:
    """ fasta:
        The fasta file should be structured such that for each sequence, there is a meta-data
        header line, beginning with a > followed immediately by the sequence identifier 
        (protein name). Additional meta-data parameters are seperated by |. If any meta-data
        parameters are passed, the second parameter (following the identifier) is expected to
        be the kingdom. 
        
        The kingdoms can be "Animal", "Plant", "Fungi" or "Other". If no kingdom is specified
        then it is expected that 
        
        jjaa-style:
        The file data should be in sets of three lines. 
        The first should contain the protein name, kingdom name and partition number, 
        separated by |. The kingdom name should be defined in the KINGDOM_ENCODING dictionary.
        The second line should contain a sequence of characters, separated by spaces.
        The third should contain a sequence of numbers, also separated by spaces. 
        
        TODO: Split filetypes into separate functions.
        """
    data = []
    def _fasta_addrow(row):
        """ Helper function for fasta parsing. Call this after all sequence
            parts have been concatenated.
        """
        ## If no identifier, then we ignore
        ## TODO: inform of a failed entry?
        if 'name' not in row:
            return
        same_but_longer = make_input_101(' '.join(row['sequence'][-100:])) 
        encoded_aminos = list(encode_aminos(same_but_longer))
        row['aa_string'] = same_but_longer
        row['enc_input'] = encoded_aminos
        seq_length = len(row['sequence'][-100:])+1
        row['seq_length'] = seq_length
        if 'output' not in row or row['output'] == None:
            row['output'] = 0
        data.append(row)
    if file_type == 'jjaa-style':
        with open('../../data/short_sequences/%s_data.txt' % (fetching)) as inf:
            row = {}
            for i,line in enumerate(inf):
                if i%3==0:
                    line_spl = line.strip().split('|')
                    row['name'] = line_spl[0]
                    row['kingdom'] = line_spl[1]
                    row['enc_kingdom'] = KINGDOM_ENCODING[line_spl[1]]
                    row['partition'] = int(line_spl[2].split("_")[-1])
                elif i%3==1:
                    same_but_longer = make_input_101(line.strip())
                    encoded_aminos = list(encode_aminos(same_but_longer))
                    row['aa_string'] = same_but_longer
                    row['sequence'] = ''.join(line.strip().split())
                    row['enc_input'] = encoded_aminos
                    seq_length = len(line.strip().split())+1
                    row['seq_length'] = seq_length
                else:
                    row['output'] = make_output_101_class_index(line.strip()) 
                    data.append(row)
                    row = {}
    elif file_type == 'fasta':
        with open(fetching) as inf:
            row = {}
            for i,line in enumerate(inf):
                stripped_line = line.strip()
                if line[0] == '>':
                    # We've encountered a header line, which means, new
                    # sequence!
                    if 'sequence' in row:
                        # Add last parsed sequences data to collection 
                        _fasta_addrow(row)
                    row = {} # reset dict
                    seq_meta = stripped_line[1:].split('|')
                    row['name'] = seq_meta[0].split()[0].strip()
                  
                    # Get kingdom
                    row['kingdom'] = None
                    row['GPI-anchored'] = None
                    row['right-pos'] = None
                    row['left-pos'] = None
                    row['output'] = None
                    row['partition'] = None
                    row['exp'] = None

                    for meta in seq_meta[1:]:
                        ms = meta.split('=')
                        if len(ms) != 2:
                            continue
                        if ms[0] == 'kingdom':
                            row['kingdom'] = ms[1]
                        elif ms[0] == 'Kingdom':
                            row['kingdom'] = ms[1]
                        elif ms[0] == 'GPI-anchored':
                            row['GPI-anchored'] = 1 == int(ms[1])
                        elif ms[0] == 'right-pos':
                            row['right-pos'] = int(ms[1])
                        elif ms[0] == 'left-pos':
                            row['left-pos'] = int(ms[1])
                            row['output'] = row['left-pos']-1
                        elif ms[0] == 'partition':
                            row['partition'] = int(ms[1])
                        elif ms[0] == 'exp':
                            row['exp'] = int(ms[1])

                    if kingdom_override != None:
                        row['kingdom'] = kingdom_override
                    if row['kingdom'] in KINGDOM_ENCODING:
                        row['enc_kingdom'] = KINGDOM_ENCODING[row['kingdom']]
                    else:
                        row['enc_kingdom'] = -1
                else:
                    # Parse sequences parts, they can be in multiple lines
                    # And can be in white space separated chunks.
                    if not 'sequence' in row:
                        row['sequence'] = ''.join(stripped_line.split())
                    else:
                        row['sequence'] += ''.join(stripped_line.split())
            # Add the last sequence in the input file
            if 'sequence' in row:
                _fasta_addrow(row)
    return data


def fetch_data_as_frame(fetching: str='train_val', file_type: str='jjaa-style', kingdom_override: str=None) -> pd.core.frame.DataFrame:
    """ Calls the fetch_data function and returns the output as a 
        DataFrame, sorted, descending, by seq_length. 
        """
    df = pd.DataFrame(fetch_data(fetching, file_type, kingdom_override))
    if len(df) > 0:
        return df.sort_values(ascending=False,by=["seq_length"]).reset_index(drop=True)
    else:
        return df


def get_batch(max_size: int, desired_batch_size: int) -> Iterator[Tuple[int,int]]:
    """ max_size: The size/length of the set to be batched.
        desired_batch_size: The desired/expected batch size.
        Iterates over the range of integers [desired_batch_size,max_size] in
        increments of desired_batch. Yielding a tuple, containing the 
        previous integer, last_index and the current. last_index is 
        initially 0.  """
    last_index = 0
    for i in range(desired_batch_size,max_size,desired_batch_size):
        yield  (last_index,i)
        last_index = i
    if last_index < max_size:
        yield (last_index,max_size)

def binarize_presence(valid_out,last_val_pred,valid_lengths):
    confusion_hasomega = {
        "tp":0, "fp":0, "tn":0, "fn":0
    }
    binarization_hasomega = []
    for true,pred,length in zip(valid_out,last_val_pred,valid_lengths):
        if pred>=(length-1) and true==(length-1):
            confusion_hasomega["tn"] += 1
            binarization_hasomega.append((0, 0))
        elif pred<(length-1) and true!=(length-1):
            confusion_hasomega["tp"] += 1
            binarization_hasomega.append((1, 1))
        elif pred>=(length-1) and true!=(length-1):
            confusion_hasomega["fn"] += 1
            binarization_hasomega.append((1, 0))
        elif pred<(length-1) and true==(length-1):
            confusion_hasomega["fp"] += 1
            binarization_hasomega.append((0, 1))
    t, p = zip(*binarization_hasomega)
    return confusion_hasomega, matthews_corrcoef(t, p)
