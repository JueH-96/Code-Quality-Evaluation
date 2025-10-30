from typing import List
from collections import defaultdict

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Create a trie from the words
        trie = {}
        for word in words:
            node = trie
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['$'] = True

        # Function to find the minimum number of valid strings to form a prefix
        def min_valid_strings(trie_node, prefix):
            if '$' in trie_node:
                return 1

            min_count = float('inf')
            for char, next_node in trie_node.items():
                if char != '$':
                    min_count = min(min_count, min_valid_strings(next_node, prefix + char))
            return min_count + 1 if min_count != float('inf') else float('inf')

        # Find the minimum number of valid strings to form the target
        min_count = float('inf')
        node = trie
        for i, char in enumerate(target):
            if char not in node:
                return -1
            node = node[char]
            min_count = min(min_count, min_valid_strings(node, target[:i+1]))
        return min_count if min_count != float('inf') else -1