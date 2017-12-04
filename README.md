# Decision-Trees-in-pure-python

dependencies : PILLOW
tutorial sources : https://sites.google.com/site/nttrungmtwiki/home/it/machine-lear/decision-tree---boosted-tree---random-forest/-decisiontree-building-a-decision-tree-from-scratch---a-beginner-tutorial


mainly to keep track of my personal study on building dtrees and rf from scratch.
trees here are built as class that are built recursively. many other implementations use dictionaries, which have the added bonus of being exportable into json.

classes can't be pickled (ikr?) so the current workaround is to instance a DecisionNode class and populate it with the tree from the saved file.
