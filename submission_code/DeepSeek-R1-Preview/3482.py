from typing import List
from collections import deque

class TrieNode:
    __slots__ = ['children', 'fail', 'output']
    def __init__(self):
        self.children = dict()
        self.fail = None
        self.output = []  # list of tuples (length, cost)

def build_trie(unique_words_with_min_cost):
    root = TrieNode()
    for word, cost in unique_words_with_min_cost.items():
        node = root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.output.append((len(word), cost))
    return root

def build_failure_links(root):
    queue = deque()
    root.fail = None
    queue.append(root)
    while queue:
        current_node = queue.popleft()
        for c, child in current_node.children.items():
            if current_node == root:
                child.fail = root
            else:
                p = current_node.fail
                while p is not None:
                    if c in p.children:
                        child.fail = p.children[c]
                        break
                    p = p.fail
                if p is None:
                    child.fail = root
            queue.append(child)
    return root

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        if not target:
            return 0
        word_to_min_cost = {}
        for word, cost in zip(words, costs):
            if word in word_to_min_cost:
                if cost < word_to_min_cost[word]:
                    word_to_min_cost[word] = cost
            else:
                word_to_min_cost[word] = cost
        if not word_to_min_cost:
            return -1
        root = build_trie(word_to_min_cost)
        root = build_failure_links(root)
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        current_node = root
        for j in range(n):
            c = target[j]
            while current_node is not None and c not in current_node.children:
                current_node = current_node.fail
            if current_node is None:
                current_node = root
            else:
                current_node = current_node.children.get(c, root)
            temp_node = current_node
            while temp_node != root:
                for length, cost in temp_node.output:
                    start = j - length + 1
                    if start >= 0:
                        if dp[start] + cost < dp[j + 1]:
                            dp[j + 1] = dp[start] + cost
                temp_node = temp_node.fail
        return dp[n] if dp[n] != float('inf') else -1