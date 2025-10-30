from typing import List

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        # Convert integers to strings
        arr1_str = [str(num) for num in arr1]
        arr2_str = [str(num) for num in arr2]
        
        # Define TrieNode
        class TrieNode:
            def __init__(self):
                self.children = [None] * 10  # For digits 0-9
        
        # Build trie with arr2
        root = TrieNode()
        for word in arr2_str:
            node = root
            for char in word:
                digit = int(char)
                if node.children[digit] is None:
                    node.children[digit] = TrieNode()
                node = node.children[digit]
        
        # Traverse with each word in arr1 and find max length of common prefix
        max_length = 0
        for word in arr1_str:
            node = root
            current_length = 0
            for char in word:
                digit = int(char)
                if node.children[digit] is None:
                    break  # Cannot go further
                node = node.children[digit]
                current_length += 1
            # Update max_length with the length found for this word
            if current_length > max_length:
                max_length = current_length
        
        return max_length