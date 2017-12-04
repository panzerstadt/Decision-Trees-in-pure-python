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
def divide_set(rows, column, value):
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
    set_1 = [row for row in rows if split_function(row)]
    set_2 = [row for row in rows if not split_function(row)]
    return set_1, set_2

# pprint(divide_set(my_data, 3, 20), indent=2)


'''
STEP 2 : ENTROPY
'''


# create counts of possible results (the last column of each row is the result)
def unique_counts(rows):
    results = {}
    for row in rows:
        r = row[len(row)-1]  # last item in the row
        if r not in results: results[r] = 0  # if key not found, make new key
        results[r] += 1  # add one count to key regardless
    return results

# pprint(unique_counts(my_data))
# divide_set(my_data, 3, 20, view_split=True)


# entropy is the sum of p(x)log(p(x)) across all
# different possible results (degree of disorder)
# entropy is the degree of disorder/randomness, so lower entropy
# means lower mixing / better division / better predictions.
def entropy(rows):
    from math import log
    log_2 = lambda x: log(x)/log(2)  # small function to make a log base 2
    results = unique_counts(rows)
    # calculate the entropy based on unique_counts
    ent = 0.0
    # for every category in unique_counts, divide number of results by
    # total rows (in the group)
    for r in results.keys():
        p = float(results[r])/len(rows)
        ent = ent - p * log_2(p)
        # entropy gets deducted every iteration
        # by the log_2(probability) weighted by proportion
    return ent


# gini index = sum of all probabilities squared
# sum of all probabilities squared = sum of each(probability of a class in relation to its set) squared
# i don't think i can use gini index here if i don't ensure that i only do binary splits
# the code only implements binary split, yes, but information gain is not measured in a gini index implementation
# TODO: find out how to implement gini index instead of entropy / IG in this code
def gini_index(rows):
    results = unique_counts(rows)
    # calculate gini_index based on unique_counts
    gini = 1 - sum([(float(results[r]) / len(rows)) ** 2 for r in results.keys()])
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


def build_tree(rows, score_function=entropy):
    """
    builds a decision tree by iterating through every column
    :param rows: dataset in nested list format
    :param score_function: what we use to split the nodes
    :return: tree as a class
    """
    # rows in the dataset, either whole dataset or part of dataset during recursion
    # score_function = impurity measurement criteria. default=entropy
    # the input is a function
    if len(rows) == 0: return decisionNode()  # len(rows) is the number of units in a set
    current_score = score_function(rows)  # the current impurity, as calculated by the score function

    # set up some variables to track the best criteria
    best_gain = 0.0
    best_criteria = None
    best_sets = None

    column_count = len(rows[0]) - 1
    # count the first row to get the num of attributes/columns
    # -1 takes out the last column which is the ylabel

    # find best gain by going through all columns and comparing impurity
    for col in range(0, column_count):
        # generate the list of all possible different values in
        # the considered column
        global column_values  # for debugging purposes
        column_values = {}

        for row in rows:
            column_values[row[col]] = 1
            # fill the dictionary with column values from each row
            # '1' is arbitrary, we just need the keys

        # now try dividing the rows up for each value in this column
        # loops through each value and calculates information gain
        # keep best gain, criteria and sets
        for value in column_values.keys():
            # the var value here is the keys in the dict
            (set1, set2) = divide_set(rows, col, value)
            # make split and put them in set1 and set2

            # information gain
            p = float(len(set1))/len(rows)
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
        return decisionNode(results=unique_counts(rows))
        # if branch is no longer 'learning'(splits don't achieve better purity),
        # return the decision node with results as properties.
        # this is the leaf


tree = build_tree(my_data)

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

print_tree(tree)


def get_width(tree):
    if tree.tb == None and tree.fb == None: return 1
    return get_width(tree.tb) + get_width(tree.fb)


def get_depth(tree):
    if tree.tb == None and tree.fb == None: return 0
    return max(get_depth(tree.tb), get_depth(tree.fb)) + 1


from PIL import Image, ImageDraw


def draw_tree(tree, jpeg = 'tree.jpg'):
    w = get_width(tree) * 100
    h = get_depth(tree) * 100 + 120

    img = Image.new('RGB', (w,h), (255,255,255))
    draw = ImageDraw.Draw(img)

    draw_node(draw, tree, w/2, 20)
    img.show(jpeg)
    img.save(jpeg, 'JPEG')


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
draw_tree(tree, jpeg='test.jpg')


'''
STEP 5 : CLASSIFY NEW OBSERVATIONS
'''
# CLASSIFY NEW OBSERVATIONS
# BACK TO THE ALGORITHM

# put a new observation into a tree
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

print('\ntesting new classification')
print('classifying [(direct), USA, no, 23]')
print('result : ', end='')
print(classify(['(direct)', 'USA', 'no', 23], tree))