from typing import List

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Initialize the root of the trie
        root = {'children': {}, 'best': (float('inf'), -1)}
        
        # Insert each word into the trie
        for i, word in enumerate(wordsContainer):
            reversed_word = word[::-1]
            current_node = root
            current_length = len(word)
            
            # Update the current node's best candidate
            if current_length < current_node['best'][0] or (current_length == current_node['best'][0] and i < current_node['best'][1]):
                current_node['best'] = (current_length, i)
            
            # Traverse through each character of the reversed word
            for c in reversed_word:
                if c not in current_node['children']:
                    current_node['children'][c] = {'children': {}, 'best': (float('inf'), -1)}
                current_node = current_node['children'][c]
                # Update the child node's best candidate
                if current_length < current_node['best'][0] or (current_length == current_node['best'][0] and i < current_node['best'][1]):
                    current_node['best'] = (current_length, i)
        
        # Process each query
        ans = []
        for query in wordsQuery:
            reversed_q = query[::-1]
            current_node = root
            best_length, best_index = root['best']
            
            # Traverse the trie with the reversed query
            for c in reversed_q:
                if c in current_node['children']:
                    current_node = current_node['children'][c]
                    best_length, best_index = current_node['best']
                else:
                    break  # No further characters match, stop here
            ans.append(best_index)
        
        return ans