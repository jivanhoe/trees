import logging
from typing import Optional, List, Tuple

import numpy as np
from graphviz import Digraph
from matplotlib.colors import is_color_like

from models.tree import Tree
from visualization.color import Color

from copy import deepcopy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreeGraph(Digraph):

    def __init__(
            self,
            tree: Tree,
            feature_names: Optional[List[str]] = None,
            class_names: Optional[List[str]] = None,
            colors: Optional[List[str]] = None,
            font: str = "verdana",
            fontsize: float = 10.0,
            max_depth_to_plot: int = 5,
            plot_on_init: bool = True
    ):
        super().__init__()
        self.tree = tree
        self.class_names = class_names
        self.feature_names = feature_names
        self.colors = colors if colors else deepcopy(TreeGraph.DEFAULT_COLORS)
        self.font = font
        self.fontsize = fontsize
        self.max_depth_to_plot = max_depth_to_plot
        if plot_on_init:
            self.plot()

    DEFAULT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:pink", "tab:cyan"]

    def _validate_colors(self):
        assert np.all([is_color_like(color) for color in self.colors]), "Error - invalid color provided."
        if self.tree.data.is_classifier:
            num_classes = self.tree.data.value.shape[0]
            num_colors = len(self.colors)
            if len(self.colors) < num_classes:
                logging.warning(f"Number of classes ({num_classes}) is greater than number of colors ({num_colors}) provided - reusing colors.")
                self.colors = [self.colors[k % num_colors] for k in range(num_classes)]

    def _get_root_node_label(self, subtree: Tree) -> str:
        if subtree.data.is_classifier:
            modal_class = np.argmax(subtree.data.value)
            modal_class_name = self.class_names[modal_class] if self.class_names else modal_class
            modal_class_pct = "{0:.0f}".format(subtree.data.value.max() * 100)
            label = f"Modal class: {modal_class_name} ({modal_class_pct}%) \n Samples: {subtree.data.n_train_samples}"
        else:
            mean_value = subtree.data.value.mean()
            pct_error = "{0:.0f}".format(np.sqrt((subtree.data.value - mean_value) ** 2) / mean_value * 100)
            label = f"Estimated value: {mean_value} (±{pct_error}%) \n Samples: {subtree.data.n_train_samples}"
        if not subtree.is_leaf():
            feature = self.feature_names[
                subtree.split_feature] if self.feature_names else f"feature {subtree.split_feature}"
            label += f"\n\n Split on: {feature}"
        return label

    @staticmethod
    def _get_split_labels(subtree: Tree) -> Tuple[str, str]:
        split_threshold = "{0:.2f}".format(subtree.split_threshold)
        return f"< {split_threshold}", f"≥ {split_threshold}"

    @staticmethod
    def _get_node_alpha(subtree: Tree) -> float:
        if subtree.data.is_classifier:
            print()
            return max(2 * subtree.data.value.max() - 1, 0)
        mean_value = subtree.data.value.mean()
        pct_error = np.sqrt((subtree.data.value - mean_value) ** 2) / mean_value
        return 1 - min(pct_error, 1)

    def _get_node_color(self, subtree: Tree) -> Color:
        color_name = self.colors[np.argmax(subtree.data.value)] if subtree.data.is_classifier else self.colors[0]
        return Color.from_string(color_name=color_name, alpha=TreeGraph._get_node_alpha(subtree=subtree))

    def _add_subtree_root_node(self, subtree: Tree) -> None:
        self.node(
            name=str(subtree),
            label=self._get_root_node_label(subtree=subtree),
            fontname=self.font,
            fontsize=str(self.fontsize),
            shape="rectangle",
            style="rounded,filled",
            fillcolor=self._get_node_color(subtree=subtree).hex_code()
        )

    def _add_parent_child_edge(self, parent: Tree, child: Tree, split_label: str) -> None:
        self.edge(
            head_name=str(child),
            tail_name=str(parent),
            label=split_label,
            fontname=self.font,
            fontsize=str(self.fontsize)
        )

    def _build_graph(self, subtree: Tree) -> None:
        if not subtree.is_leaf() and subtree.root_depth < self.max_depth_to_plot:
            for child, split_label in zip(subtree.get_children(), TreeGraph._get_split_labels(subtree=subtree)):
                self._add_subtree_root_node(subtree=child)
                self._add_parent_child_edge(parent=subtree, child=child, split_label=split_label)
                self._build_graph(subtree=child)

    def plot(self):
        self._validate_colors()
        self._add_subtree_root_node(subtree=self.tree)
        self._build_graph(self.tree)
