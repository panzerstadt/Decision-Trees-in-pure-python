# Decision-Trees-in-pure-python

dependencies : PILLOW

tutorial source 1 : https://sites.google.com/site/nttrungmtwiki/home/it/machine-lear/decision-tree---boosted-tree---random-forest/-decisiontree-building-a-decision-tree-from-scratch---a-beginner-tutorial

tutorial source 2 : https://machinelearningmastery.com/implement-random-forest-scratch-python/

mainly to keep track of my personal study on building dtrees and rf from scratch.
trees here are built as class that are built recursively. many other implementations use dictionaries, which have the added bonus of being exportable into json.

classes can't be pickled (ikr?) so the current workaround is to instance a DecisionNode class and populate it with the tree from the saved file.
