from typing import List

class TrieNode:
    def __init__(self):
        self.children = {}

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Build a trie from all numbers in arr1 (represented as strings of digits).
        root = TrieNode()
        
        def insert(num_str: str) -> None:
            node = root
            for ch in num_str:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
        
        def get_prefix_length(num_str: str) -> int:
            # Traverse trie for num_str, counting how many digits match
            node = root
            length = 0
            for ch in num_str:
                if ch in node.children:
                    node = node.children[ch]
                    length += 1
                else:
                    break
            return length
        
        # Insert all numbers from arr1 into the trie
        for num in arr1:
            insert(str(num))
        
        # For each number in arr2, find how many digits match in the trie
        # Keep track of the maximum prefix length seen.
        max_prefix_len = 0
        for num in arr2:
            prefix_len = get_prefix_length(str(num))
            if prefix_len > max_prefix_len:
                max_prefix_len = prefix_len
        
        return max_prefix_len