import numpy as np
import torch.nn.functional as F
from torch import device, from_numpy

ONE_HOT_POS_TO_SPECIAL_CHAR = {
    95: '\x02',  # SOS
    96: '\x03',  # EOS
    97: '\t'  # unknown -> null
}
REVIEW_BATCH_SIZE = 1
MAX_GENERATE_LEN = 1500
ONE_HOT_BEER_STYLE_VECTOR_LEN = 104
ONE_HOT_RATING_VECTOR_LEN = 5
ONE_HOT_CHAR_VECTOR_LEN = 98
BIAS_VECTOR_LEN = 1
ONE_HOT_VECTOR_LEN = (ONE_HOT_BEER_STYLE_VECTOR_LEN +
                      BIAS_VECTOR_LEN +
                      ONE_HOT_RATING_VECTOR_LEN +
                      ONE_HOT_CHAR_VECTOR_LEN)
CHAR_START_IDX = (ONE_HOT_BEER_STYLE_VECTOR_LEN +
                  ONE_HOT_RATING_VECTOR_LEN +
                  BIAS_VECTOR_LEN)

# convert each character to an one-hot vector sized 97 using ASCII value - 32
# exceptions: SOS = 95, EOS = 96, unknown char = 97
def char2pos(c: str) -> int:
    pos = ord(c)
    if pos == 2:  # SOS
        return 95
    if pos == 3:  # EOS
        return 96
    # normal ASCII starting at space, minus the offset 32
    pos -= ord(' ')
    if pos < 0:  # convert escape chars into spaces
        return 0
    if pos > 94:
        return 97

    return pos


def pos2char(pos: int) -> str:
    pos = int(pos)
    if pos < 95:
        # normal characters
        return chr(pos + 32)
    # special characters
    return ONE_HOT_POS_TO_SPECIAL_CHAR[pos]


def generate_once(model, X_test, temporature: float = 0.2):
    # keep track of the maximum length
    count = MAX_GENERATE_LEN
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
        pos = np.random.choice(ONE_HOT_CHAR_VECTOR_LEN, 1,
                               p=probs.detach().cpu().numpy())[0]
        if (pos == 96):  # stop of reached End of Sentence
            break
        # add current char to the review
        review_chars.append(pos2char(pos))

        # zero out the character sublist
        for i in range(ONE_HOT_CHAR_VECTOR_LEN):
            X_test[0][0][CHAR_START_IDX + i] = 0
        # set the bit for the new character
        X_test[0][0][CHAR_START_IDX + pos] = 1

        count -= 1

    return ''.join(review_chars)
