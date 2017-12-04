"""
ONE DECISION TREE

below is an implementation of a decision tree built as a class.
"""
# todo: build a decision tree as a dictionary (for easier saving and loading, and also possibility for saving into json)

print('\n', '%s' %'-'*100)
print('one decision tree')
print('%s' %'-'*100)

# source : https://sites.google.com/site/nttrungmtwiki/home/it/machine-lear/decision-tree---boosted-tree---random-forest/-decisiontree-building-a-decision-tree-from-scratch---a-beginner-tutorial
# doesn't deal with missing data
# without pruning

"""
Use Cases for data exploration : sorts most significant variables

For example, we are working on a problem where we have information available
in hundreds of variables, there decision tree will help to
identify most significant variable.
"""

from pprint import pprint

# example dataset
my_data=[['slashdot','USA','yes',18,'None'],
        ['google','France','yes',23,'Premium'],
        ['digg','USA','yes',24,'Basic'],
        ['kiwitobes','France','yes',23,'Basic'],
        ['google','UK','no',21,'Premium'],
        ['(direct)','New Zealand','no',12,'None'],
        ['(direct)','UK','no',21,'Basic'],
        ['google','USA','no',24,'Premium'],
        ['slashdot','France','yes',19,'None'],
        ['digg','USA','no',18,'None'],
        ['google','UK','no',18,'None'],
        ['kiwitobes','UK','no',19,'None'],
        ['digg','New Zealand','yes',12,'Basic'],
        ['slashdot','UK','no',21,'None'],
        ['google','UK','yes',18,'Basic'],
        ['kiwitobes','France','yes',19,'Basic']]


# this is a decorator
# gives the wrapped functions a 'view_split' keyword argument.
# decorators are meant to be reusable, but then this one isn't really..
def see_data(function_to_wrap):
    def decorated(*args_from_function, **kwargs_from_function):
        try:
            if kwargs_from_function['view_split'] == True:
                run_function = function_to_wrap(*args_from_function)

                temp_split = run_function  # just to make it clearer

                temp_1 = temp_split[0]
                pprint(temp_1)
                pprint(unique_counts(temp_1))
                print('')
                temp_2 = temp_split[1]
                pprint(temp_2)
                pprint(unique_counts(temp_2))
                return run_function
            else:
                return function_to_wrap(*args_from_function)
        except:
            return function_to_wrap(*args_from_function)

    return decorated


# steps
# 1. split dataset into children sets
# 2. entropy
# 3. build a tree recursively
# 4. represent the trees graphically
# 5. classify with the new tree

'''
STEP 1 : SPLIT DATASET INTO CHILDREN SETS
'''
# can handle numeric and nominal (categorical) values

# like gh's dispatch component
# divide_set splits dataset row by row into two based on the
# given criteria (value) on a
# given column index (column)


@see_data
def divide_set(dataset, column, value):
    split_function = None
    if isinstance(value, int) or isinstance(value, float):
        # give a yes/no based on whether the value is larger than the tested
        # value or not
        # not sure if this is actually the best way to do it
        split_function = lambda row: row[column] >= value
        # still not too sure what this does
    else:
        # gives a yes/no based on whether the value is same or different
        split_function = lambda row: row[column] == value

    # divide the rows into two sets and return them
    set_1 = [row for row in dataset if split_function(row)]
    set_2 = [row for row in dataset if not split_function(row)]
    return set_1, set_2

# pprint(divide_set(my_data, 3, 20), indent=2)


'''
STEP 2 : ENTROPY
'''


# create counts of possible results (the last column of each row is the result)
def unique_counts(dataset):
    results = {}
    for row in dataset:
        r = row[len(row)-1]  # last item in the row
        if r not in results: results[r] = 0  # if key not found, make new key
        results[r] += 1  # add one count to key regardless
    return results

# pprint(unique_counts(my_data))
# divide_set(my_data, 3, 20, view_split=True)


# entropy is the sum of p(x)log(p(x)) across all
# different possible results (degree of disorder) in the given dataset
# (can be entire set or subsets that have been split by dtree)
# entropy is the degree of disorder/randomness, so lower entropy
# means lower mixing / better division / better predictions.
def entropy(dataset):
    from math import log
    log_2 = lambda x: log(x)/log(2)  # small function to make a log base 2
    results = unique_counts(dataset)
    # calculate the entropy based on unique_counts
    ent = 0.0
    # for every category in unique_counts, divide number of results by
    # total rows (in the group)
    for r in results.keys():
        p = float(results[r])/len(dataset)
        ent = ent - p * log_2(p)
        # entropy gets deducted every iteration
        # by the log_2(probability) weighted by proportion
    return ent


# gini index = sum of all probabilities squared
# sum of all probabilities squared = sum of each(probability of a class in relation to its set) squared
# i don't think i can use gini index here if i don't ensure that i only do binary splits
# the code only implements binary split, yes, but information gain is not measured in a gini index implementation
# TODO: find out how to implement gini index instead of entropy / IG in this code
def gini_index_WIP(dataset):
    results = unique_counts(dataset)
    # calculate gini_index based on unique_counts
    gini = 1 - sum([(float(results[r]) / len(dataset)) ** 2 for r in results.keys()])
    return gini


def describe(my_data):
    print('given dataset:')
    pprint(my_data, indent=2)
    print('\nentropy(mixed-ness) of initial dataset: ', entropy(my_data))

    print('\n\ncandidate split by feature column "age" into 2 groups')
    set1, set2 = divide_set(my_data, 3, 20, view_split=True)
    print('\nentropy for set 1:', entropy(set1))
    print('entropy for set 2:', entropy(set2))

describe(my_data)

'''
STEP 3 : BUILD A DECISION TREE RECURSIVELY
'''
# tldr: calculate information gain per feature split and compare to find
# the highest, then split it, and do the same thing for each children node

# information gain = differece between the current entropy and
# weighted average entropy of two new groups


class decisionNode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col  # column index of criteria to be tested (index 1 = country)
        self.value = value  # the value to match (according to column) to split datset
        self.results = results
        self.tb = tb  # decision node
        self.fb = fb


def build_tree(dataset, score_function=entropy, max_depth=0, min_size=1):
    """
    builds a decision tree by iterating through every column
    :param dataset: dataset in nested list format
    :param score_function: what we use to split the nodes
    :return: tree as a class
    """
    # rows in the dataset, either whole dataset or part of dataset during recursion
    # score_function = impurity measurement criteria. default=entropy
    # the input is a function
    if len(dataset) == 0: return decisionNode()  # len(dataset) is the number of units in a set
    current_score = score_function(dataset)  # the current impurity, as calculated by the score function

    # set up some variables to track the best criteria
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(dataset[0]) - 1
    # count the first row to get the num of attributes/columns
    # -1 takes out the last column which is the ylabel

    # find best gain by going through all columns and comparing impurity
    for col in range(0, column_count):
        # generate the list of all possible different values in
        # the considered column
        global column_values  # for debugging purposes
        column_values = {}

        for row in dataset:
            column_values[row[col]] = 1
            # fill the dictionary with column values from each row
            # '1' is arbitrary, we just need the keys

        # now try dividing the rows up for each value in this column
        # loops through each value and calculates information gain
        # keep best gain, criteria and sets
        for value in column_values.keys():
            # the var value here is the keys in the dict
            (set1, set2) = divide_set(dataset, col, value)
            # make split and put them in set1 and set2

            # information gain
            p = float(len(set1))/len(dataset)
            # p is the size of a child set relative to its parent
            # why calculate p? because it is used as the weight multiplier
            # for information gain (below)

            # calculate how much information we gain from splitting the
            # parent node into this particular set of child nodes
            gain = current_score - (p * score_function(set1)) - ((1 - p) * score_function(set2))
            # cf. formula information gain (what is cf?)
            # current score is the entropy of the node before splitting
            '''
            formula for IG:
            IG(btwn parent and children) = entropy_parent - (entropy_child1 * proportion_child1) -
            (entropy_child2 * proportion_child2)
            
            information gained by splitting the dataset by this* feature is calculated
            by taking the entropy(messiness) of the parent node and subtracting the entropy
            of both children weighted by the number of rows they represent (so if the messiness
            of both children add up to a higher entropy, it would be considered information loss,
            and would not be used).
            
            *this being the current column in the iteration
            '''

            # if set is not empty, and the gain is improving over the previous
            # measure of impurity, make this new gain the best gain
            if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                # set must not be empty
                best_gain = gain
                best_criteria = (col, value)  # remember, value is column_values.keys()
                best_sets = (set1, set2)

    # make branch according to the split that makes gives the best gain
    if best_gain > 0:
        # make sub branches
        # by calling the same definition (recursion)
        trueBranch = build_tree(best_sets[0])
        falseBranch = build_tree(best_sets[1])
        return decisionNode(col=best_criteria[0],
                            value=best_criteria[1],
                            tb=trueBranch,
                            fb=falseBranch)
    else:
        return decisionNode(results=unique_counts(dataset))
        # if branch is no longer 'learning'(splits don't achieve better purity),
        # return the decision node with results as properties.
        # this is the leaf. current implementation splits until each node is 100% pure

def save_tree(tree_class, name='saved_tree'):
    import pickle
    file = open(name + '.txt', 'wb')
    file.write(pickle.dumps(tree_class.__dict__))
    file.close()

def load_tree(tree_name):
    import pickle
    temp = decisionNode()
    file = open(tree_name + '.txt', 'rb')
    dataPickle = file.read()
    file.close()

    temp.__dict__ = pickle.loads(dataPickle)
    return temp


#tree = build_tree(my_data)
#save_tree(tree)

# if the tree has been built before, load tree
#tree = load_tree('tree')



'''
STEP 4 : DISPLAYING TREES
(DETOUR FROM THE MAIN ALGORITHM
'''

def print_tree(tree, indent=''):
    # is this a leaf node?
    if tree.results != None:
        print(str(tree.results))
    else:
        print(str(tree.col) + ':' + str(tree.value) + '? ')
        # print the branches
        print(indent + 'T->', end=" ")
        print_tree(tree.tb, indent + '   ')
        print(indent + 'F->', end=" ")
        print_tree(tree.fb, indent + '   ')

# print('\nshowing grown tree:\n')
# print_tree(tree)


def get_width(tree):
    if tree.tb == None and tree.fb == None: return 1
    return get_width(tree.tb) + get_width(tree.fb)


def get_depth(tree):
    if tree.tb == None and tree.fb == None: return 0
    return max(get_depth(tree.tb), get_depth(tree.fb)) + 1


from PIL import Image, ImageDraw


def draw_tree(tree, jpeg='tree'):
    w = get_width(tree) * 100
    h = get_depth(tree) * 100 + 120

    img = Image.new('RGB', (w,h), (255,255,255))
    draw = ImageDraw.Draw(img)

    draw_node(draw, tree, w/2, 20)
    img.save(jpeg + '.jpg', 'JPEG')


def draw_node(draw, tree, x, y):
    if tree.results == None:
        # get the width of each branch
        w1 = get_width(tree.fb) * 100
        w2 = get_width(tree.tb) * 100

        # determine the total space required by this node
        left = x - (w1 + w2)/2
        right = x + (w1 + w2)/2

        # draw the condition string
        draw.text((x-20, y-10), str(tree.col) + ':' + str(tree.value), (0,0,0))

        # draw links to the branches
        draw.line((x,y,left+w1/2, y+100), fill=(255,0,0))
        draw.line((x,y,right-w2/2, y+100), fill=(255,0,0))

        # draw the branch nodes
        draw_node(draw, tree.fb, left+w1/2, y+100)
        draw_node(draw, tree.tb, right-w2/2, y+100)
    else:
        txt= ' \n'.join(['%s:%d' %v for v in tree.results.items()])
        draw.text((x-20, y), txt, (0,0,0))

# saves into jpeg
# draw_tree(tree, jpeg='test.jpg')


'''
STEP 5 : CLASSIFY NEW OBSERVATIONS
'''
# CLASSIFY NEW OBSERVATIONS
# BACK TO THE ALGORITHM

# put a new observation into a tree
# equivalent to making a prediction
def classify(observation, tree):
    """
    think of all the if statements in recursive algorithms as
    final outputters. the main code (in the else) is run until an
    if statement is achieved.
    :param observation: the new input to test on the tree
    :param tree: the pretrained model / pregrown tree
    :return: predictions
    """

    # if the current branch is a leaf node, give results and end.
    if tree.results != None:
        return tree.results
    else:
        # for the given observation, pick the relevant column in the first
        # node and test its value. there are two ways to handle them,
        # integer/floats (quantitative data) or names/labels (qualitative data)
        # tree.col is an index number stored in the tree when building it
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            # for numbers (int of float), if higher, goes into true branch
            # if false, goes into false branch
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            # for non numbers, if same value, it goes down the true branch
            # if false, goes down the false branch
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)

# print('\ntesting new classification')
# print(classify(['(direct)', 'USA', 'no', 23], tree))



"""
MANY TREES REPEATED AND AVERAGED, 
WITH SAMPLES TAKEN INTO ACCOUNT, 
THEN PUT BACK INTO THE POOL (OF SAMPLES). 

ALSO CALLED BOOTSTRAP AGGREGATION / BAGGING (BAGGED TREES)
SAMPLING METHOD IS CALLED SAMPLING WITH REPLACEMENT.
"""

print('\n', '%s' %'-'*100)
print('bagged decision trees')
print('%s' %'-'*100)

# Bagged decision trees cannot overfit their problem
# Trees can be added until maximum performance is achieved.

# sonar dataset used
dataset_filepath = './UCI/sonar.all-data.txt'

#print(open(dataset_filepath).read())

'''
STEP 1 : MAKE A SUBSAMPLE OF THE DATASET
randomly select rows, add them to new list until a ratio is
reached
'''

def subsample(dataset, ratio=1.0):
    from random import randrange

    sample = []
    num_of_samples = round(ratio*len(dataset))
    while len(sample) < num_of_samples:
        # with replacement, meaning the random index generator
        # can repeat numbers
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def mean(numbers):
    return sum(numbers) / float(len(numbers))

from random import seed, randrange


print('toy example showing that the larger the amount of samples the more accurate the model:\n')
seed(1)
dataset = [[randrange(10)] for i in range(20)]
print('True Mean: %s' %mean([row[0] for row in dataset]))

ratio = 0.10
for size in [1, 10, 100]:
    sample_means = []
    for i in range(size):
        sample = subsample(dataset, ratio=ratio)
        sample_mean = mean([row[0] for row in sample])
        sample_means.append(sample_mean)
    print('means of each sample', sample_means)
    print('averaged mean', mean(sample_means))
print('\n** bagging means that instead of only calculating the mean,\n'
      'we can create a model for each subsample')

'''
STEP 2 : PREPROCESSING
'''

def load_csv(filename):
    from csv import reader
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def numerical_str_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
        # swapping values directly in the list

def categorical_str_to_label(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)  # finds how many similar. set() works like cset in gh
    # set() returns only unique instances in a list
    lookup = {}
    for i, value in enumerate(unique):
        lookup[value] = i  # make a lookup table with keywords as dict keys and class values as values
    for row in dataset:
        row[column] = lookup[row[column]]  # swaps values in dataset with values from lookup table
    return lookup

def cross_validation_split(dataset, n_folds):
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))  # a random index number
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        # for very fold (without replacement), run dtree/baging, get results,
        # calculate accuracy, throw it in a list
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])  # sum while starting from an empty list '[]' instead of 0
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        # container for different algorithms to test
        # e.g. bagging. dtree algorithm can be written to conform to this format,
        # allowing for evaluate_algorithm to work with it
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def gini_index(groups, classes):
    '''
    gini index = weighted probability score (sum of all probabilities in the subset)
    gini index = sum of all probabilities squared
    sum of all probabilities squared = sum of each(probability of a class in relation to its set) squared

    # source : https://machinelearningmastery.com/implement-bagging-scratch-python/
    # the code format isn't as easy to understand as the one by
    # this source : https://sites.google.com/site/nttrungmtwiki/home/it/machine-lear/decision-tree---boosted-tree---random-forest/-decisiontree-building-a-decision-tree-from-scratch---a-beginner-tutorial
    # so should be rewritten to read more like the formula

    groups in this case refer to nodes (sub nodes that have undergone binary split)
    :param groups: parent / child node(s). any nodes actually
    :param classes: entire set of classes (in the full dataset, not only the subset)
    :return:
    '''

    # count all samples at split points
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p  # probabilities squared
        # weight the group score (iteratively sum the probabilities) by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

def bagging_predict(trees, test_row):
    predictions = [classify(test_row, tree) for tree in trees]
    predictions = [list(i.keys())[0] for i in predictions]
    # right.. how does this return work?
    # for all trees that were given (built before), make predictions
    # and return the prediction with the highest occurence in the list
    # (tree voting)
    return max(set(predictions), key=predictions.count)

def bagging(train, test, max_depth, min_size, sample_size, n_trees):
    '''
    build tree here isn't from tutorial, but from previous decision tree tutorial.
    as such, the tree has no max_depth and min_size cutoff
    :param train:
    :param test:
    :param max_depth: not implemented
    :param min_size: not implemented
    :param sample_size: equivalent to ratio
    :param n_trees:
    :return:
    '''
    trees = []
    for i in range(n_trees):
        # for every count in number of trees to use...
        sample = subsample(train, ratio=sample_size)  # ...make subsample (with replacement)
        tree = build_tree(sample, max_depth=max_depth, min_size=min_size)  # build that tree based on subsample
        # save_tree(tree, name='tree_' + str(i))
        # draw_tree(tree, jpeg='tree_'+ str(i))
        trees.append(tree)  # put that tree in a bag/list

    # for every row (observations/input) in test set,
    # test the full bag on the row and return prediction
    predictions = [bagging_predict(trees, row) for row in test]
    return(predictions) # return predictions as a tuple


'''
BUILD / LOAD THE TREE
'''

seed(1)
n_folds = 5
max_depth = 6  # not implemented
min_size = 2  # not implemented
sample_size = 0.50
sonar_dataset = load_csv(dataset_filepath)

n_trees = 10
bagged_tree_accuracy = evaluate_algorithm(sonar_dataset, bagging, n_folds, max_depth, min_size, sample_size, n_trees)

print('bagged tree accuracy on sonar dataset')
print(bagged_tree_accuracy)

#sonar_tree = build_tree(test_dataset)
#save_tree(sonar_tree)
#sonar_tree = load_tree('sonar_tree')
#print('dtree "sonar tree" loaded')

#print_tree(sonar_tree)
#draw_tree(sonar_tree, 'sonar.jpg')

