from sklearn.tree import export_graphviz
from subprocess import Popen
from PIL import Image


def sort_features(feature_names, feature_importances):
    """Sorts features by importance.

    Arguments:
        feature_names(list): A list of strings containing the names
                             of the features.
        feature_importances(list): A list of the importances of the features.

    Returns:
        A list of (feature_name, feature_importance) tuples sorted in order
        of descending importance.
    """

    return sorted(zip(feature_names,
                      feature_importances),
                  key=lambda x: x[1],
                  reverse=True)

def display_decision_tree(tree, feature_names, class_names):
    """Displays a decision tree using graphviz.

    Arguments:
        tree(object): A trained sklearn DecisionTreeClassifier.
        feature_names(list): A list of the names of the features.
        class_names(list): A list of the names of the classes.
    """

    export_graphviz(decision_tree=tree,
                    out_file="tree.dot",
                    feature_names=feature_names,
                    class_names=class_names,
                    filled=True,
                    rounded=True)

    Popen(['dot', '-Tpng', "tree.dot", '-o', "tree.png"]).wait()
    img = Image.open('tree.png')
    img.show()
