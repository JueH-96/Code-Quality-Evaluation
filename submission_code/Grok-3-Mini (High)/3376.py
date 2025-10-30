from typing import List

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.min_len = float('inf')
                self.min_index = -1
        
        root = TrieNode()
        
        # Insert all reversed words into trie with their lengths and indices
        for index, word in enumerate(wordsContainer):
            rev_word = word[::-1]
            len_word = len(word)
            node = root
            
            # Update the current node starting from root
            if len_word < node.min_len or (len_word == node.min_len and index < node.min_index):
                node.min_len = len_word
                node.min_index = index
            
            # Traverse each character in the reversed word and update nodes
            for char in rev_word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                # Update the child node with the word's length and index
                if len_word < node.min_len or (len_word == node.min_len and index < node.min_index):
                    node.min_len = len_word
                    node.min_index = index
        
        # Process each query and find the best index
        ans = []
        for query in wordsQuery:
            rev_query = query[::-1]
            curr_node = root
            
            # Traverse the trie with the reversed query until mismatch
            for char in rev_query:
                if char in curr_node.children:
                    curr_node = curr_node.children[char]
                else:
                    break  # Stop at the point of mismatch
            
            # The current node has the precomputed min_index for the subtree
            ans.append(curr_node.min_index)
        
        return ans