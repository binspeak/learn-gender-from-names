from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from random import shuffle
import time, io, sys, json

# change these values to adjust learning and output

# the model input are ordinals of the characters in the name. It picks num_chars from
# the beginning and the end of the name (might overlap). So if this number is 3,
# the network has 6 input neurons (+ bias) and the first and last 3 characters are considered.
# Names with length < num_chars are ignored during learning
num_chars = 3

# maximal number of learning iterations. If the trainer converges before, less iterations are made
max_epochs = 5

# size of the hidden layer for the used neural network
hidden_layer_size = 4

# The set of names will be split into training and validation data. A value of 0.1 means, that 10 percent are used
# for training, 90 percent for validation
training_set_percentage = 0.5

#set this to true to use stored network weights instead of learning (data/weights.json)
use_stored_weights = False

def readin(filename, is_female):
    '''
    reads in a file with one name per line
    each entry in the resulting list includes num_chars*2 integers, which stand for ord(<character>)-96
    the last entry is used as the output and defines if the name is female (1) or male (0)
    e.g.: the name 'abcdef' would return [1,2,3,4,5,6,1] if is_female is set to 1
    '''
    with io.open(filename, encoding='utf8') as f:
        file_content = f.read()
    file_content = file_content.lower()
    names = file_content.split(",")
    result = []
    for name in names:
        if (len(name) >= num_chars):
            try:
                result.append([name_to_ord(name) , is_female])
            except ValueError as ex:
                continue # just ignore names, which do not fit
    return result


def name_to_ord(name):
    '''
    transform a given name into ordinal representation
    :param name: a given name (Note: must have at least num_chars chars)
    :return: a list of ordinals with length num_chars*2
    '''
    ordinals = [(ord(c)-96) for c in name]
    start_chars = ordinals[0:num_chars]
    end_chars = ordinals[(len(ordinals) - num_chars):len(ordinals)]
    return start_chars + end_chars

def validate_name(name):
    '''
    returns the suggestion of the neural network.
    :param name: the name to be checked
    :return: 'male' or 'female'
    '''
    prob = net.activate(name_to_ord(name.lower()))
    return 'male' if prob < 0.5 else 'female'

def validate_ord(ordinals, is_female, net):
    '''
    activates the given net with the given ordinals and returns true, if the outcome fits to the is_female parameter
    (i.e. if activation > 0.5 and is_female=1, returns true ... )
    :param ordinals: a list of ordinals which represent the chars of a name. Can be generated ny name_to_ord)
    :param is_female: 1 if the given name is supposed to be female, 0 otherwise
    :param net: the neural network instance
    :return: True is outcome fits expectation, False otherwise
    '''
    prob = net.activate(ordinals)
    return True if (prob > 0.5 and is_female==1) or (prob <= 0.5 and is_female==0) else False

def validate_dataset(dataset, net):
    '''
    Validates a given dataset and returns stats about the validation
    :param dataset: must be an iteratable which iterates over lists with two elements: ords and is_female
    :param net: the neural network instance
    :return: statistics about the validation
    '''
    hits = fails = 0
    start = time.time()
    for params, is_female in dataset:
        if (len(params) >= num_chars):
            if (validate_ord(params, is_female, net)):
                hits += 1
            else:
                fails += 1
    return {'length': len(dataset),
            'hits': hits,
            'fails': fails,
            'ratio': hits / (fails + hits),
            'time': (time.time() - start)
            }

# main execution
if __name__ == "__main__":
    # initialize the used network and dataset
    net = buildNetwork(num_chars*2, hidden_layer_size, 1, bias=True)
    ds = SupervisedDataSet(num_chars*2, 1)

    # readin names from files
    result = readin('data/female.txt', 1) + readin('data/male.txt', 0)
    shuffle(result)

    #add entries to the dataset
    for entry in result:
        ds.addSample(entry[0], (entry[1]))

    leftDs, rightDs = ds.splitWithProportion(training_set_percentage)

    if (use_stored_weights == False):
        print("successs rate over all names before training:")
        print(validate_dataset(result,net))
        print()
        #train the network
        print("starting to train the network")


        trainer = BackpropTrainer(net, leftDs)
        trainer.trainUntilConvergence(maxEpochs=max_epochs)
        #store learned parameters to file

        print("validate against the validation dataset")
        print(validate_dataset(rightDs, net))
        print()

        with io.open('data/weights.json', 'w', encoding='utf8') as f:
            f.write(json.dumps(net.params.tolist()))

    else:
        with io.open('data/weights.json') as f:
            weights = f.read()
        weights = json.loads(weights)
        net._setParameters(weights)

    print("successs rate over all names with loaded/learned weights:")
    print(validate_dataset(result, net))

    if (len(sys.argv) > 1):
        for name in sys.argv[1:]:
            print(name,'is a',validate_name(name),'name')
