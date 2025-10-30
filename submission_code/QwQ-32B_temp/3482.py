from typing import List
from collections import deque

class Node:
    __slots__ = ['children', 'fail', 'cost', 'length']
    def __init__(self):
        self.children = {}
        self.fail = None
        self.cost = None
        self.length = 0  # length from root to this node

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Preprocess words to get minimal cost for each unique word
        word_cost = {}
        for word, cost in zip(words, costs):
            if word not in word_cost or cost < word_cost[word]:
                word_cost[word] = cost
        
        # Build the trie
        root = Node()
        for word, cost in word_cost.items():
            node = root
            for c in word:
                if c not in node.children:
                    node.children[c] = Node()
                    node.children[c].length = node.length + 1
                node = node.children[c]
            node.cost = cost  # set the cost at the end node
        
        # Build failure links using BFS
        queue = deque()
        root.fail = None
        for child in root.children.values():
            child.fail = root
            queue.append(child)
        
        while queue:
            current_node = queue.popleft()
            for char, child in current_node.children.items():
                # Find the fail node for child
                fail_node = current_node.fail
                while fail_node is not None and char not in fail_node.children:
                    fail_node = fail_node.fail
                if fail_node is None:
                    child.fail = root
                else:
                    child.fail = fail_node.children[char]
                queue.append(child)
        
        # Initialize DP array
        n = len(target)
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0  # base case: empty string
        
        current_node = root
        for i in range(n):
            c = target[i]
            # Move to the correct node via failure links
            while current_node != root and c not in current_node.children:
                current_node = current_node.fail
            # Now try to move to the child
            if c in current_node.children:
                current_node = current_node.children[c]
            else:
                current_node = root  # reset to root if no path
            
            # Process all possible words ending at this node or via failure links
            temp_node = current_node
            while temp_node is not None:
                if temp_node.cost is not None:
                    l = temp_node.length
                    start = i - l + 1
                    if start >= 0:
                        if dp[start] + temp_node.cost < dp[i + 1]:
                            dp[i + 1] = dp[start] + temp_node.cost
                temp_node = temp_node.fail  # follow failure links
        
        return dp[n] if dp[n] != INF else -1