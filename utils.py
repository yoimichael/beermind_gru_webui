import constants
import numpy as np
import torch.nn.functional as F
from torch import device, from_numpy


ONE_HOT_POS_TO_SPECIAL_CHAR = {
    constants.SOS_VEC_POS: '\x02',  # SOS
    constants.EOS_VEC_POS: '\x03',  # EOS
    constants.UNKNOWN_CHAR_VEC_POS: '\t'  # unknown -> null
}


def char2pos(c: str) -> int:
    pos = ord(c)
    if pos == 2:  # SOS
        return constants.SOS_VEC_POS
    if pos == 3:  # EOS
        return constants.EOS_VEC_POS
    # normal ASCII starting at space, minus the offset 32
    pos -= ord(' ')
    if pos < 0:  # convert escape chars into spaces
        return 0
    if pos > 94:
        return constants.UNKNOWN_CHAR_VEC_POS
    return pos


def pos2char(pos: int) -> str:
    pos = int(pos)
    if pos < constants.SOS_VEC_POS:
        # normal characters
        return chr(pos + ord(' '))
    # special characters
    return ONE_HOT_POS_TO_SPECIAL_CHAR[pos]


def generate_once(model, X_test, temporature: float = 0.2):
    # keep track of the maximum length
    count = constants.MAX_GENERATE_LEN
    review_chars = []
    # let the model generate next characters
    while(count > 0):
        # get the next output from the model
        output = model(from_numpy(X_test).float().to(device("cpu")))
        # if use temperature when testing
        output /= temporature
        # find probability of each character
        output = F.softmax(output, dim=2)
        probs = output[0][0]
        # find the character
        pos = np.random.choice(constants.ONE_HOT_CHAR_VECTOR_LEN, 1,
                               p=probs.detach().cpu().numpy())[0]
        if (pos == constants.EOS_VEC_POS):  # stop of reached End of Sentence
            break
        # add current char to the review
        review_chars.append(pos2char(pos))

        # zero out the character sublist
        for i in range(constants.ONE_HOT_CHAR_VECTOR_LEN):
            X_test[0][0][constants.CHAR_START_IDX + i] = 0
        # set the bit for the new character
        X_test[0][0][constants.CHAR_START_IDX + pos] = 1

        count -= 1

    return ''.join(review_chars)
