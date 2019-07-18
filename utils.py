import numpy as np
import torch.nn.functional as F
from torch import device, from_numpy

max_len = 1500
# convert each character to an one-hot vector sized 97 using ASCII value - 32
# exceptions: SOS = 95, EOS = 96, unknown char = 97
def char2pos(c):
    pos = ord(c)
    # if it's SOS or EOS
    if (pos == 2):
        pos = 95
    elif (pos == 3):
        pos = 96
    else:
        # other normal ASCII
        pos -= 32
        if pos < 0: 
            # make the \n\t a space
            pos = 0
        elif pos > 94:
            pos = 97

    return pos

def char2oh(c):
    ans = np.zeros(98)
    ans[char2pos(c)] = 1
    return ans

def pos2char(pos):
    pos = int(pos)
    
    # special characters
    if (pos == 95): 
        return '\x02' #SOS
    elif (pos == 96):
        return '\x03' #EOS
    elif (pos == 97):
        return '\t' #unknown -> null 

    # normal characters
    return chr(pos + 32)

def oh2char(oh):
    # find which one
    pos = oh.argmax()

    return pos2char(pos)
def generate(model, X_test, temporature=0.2):
    # Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.        
        
    computing_device = device("cpu")
     
    batchSize = len(X_test)
    reviews = [""] * batchSize
    
     # prepare the model
    model.zero_grad()
    # each time we are only generating one character
    model.hidden = model.init_hidden(batchSize)
    
    # keep track of the maximum length
    count = max_len
    
    # let the model generate next characters
    while(count > 0):

        # get the next output from the model
        output = model(from_numpy(X_test).float().to(computing_device))

        # if use temperature when testing
        
        output /= temporature

        # find probability of each character
        output = F.softmax(output,dim=2)
        
        # check if all are done
        eoses = 0
        
        for i in range(batchSize):
            probs = output[i][0]
            # find the character 
            pos = np.random.choice(98, 1, p=probs.detach().cpu().numpy())[0]
            
            if (pos == 96):
                eoses += 1
            # add current char to the review
            reviews[i] += pos2char(pos)
                        
            # update X_test
            oh = np.zeros(98)
            oh[pos] = 1
            X_test[i][0][-98:] = oh
            del oh

        # if all are done
        if (eoses == batchSize):
            break

        count -= 1
    
    return reviews
