from typing import List

class TrieNode:
    def __init__(self):
        # A dictionary to store children nodes, mapping character to TrieNode
        self.children = {}
        # Stores (min_length, original_index) of the word from wordsContainer
        # that, when reversed, forms a prefix ending at this node and satisfies
        # the criteria:
        # 1. Smallest original string length
        # 2. Smallest original index (for tie-breaking on length)
        # Initialize with infinity length and an invalid index (-1)
        self.best_candidate_info = (float('inf'), -1)

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        
        root = TrieNode()
        
        # Build the Trie with reversed words from wordsContainer
        # At each node, store the best candidate based on length and original index.
        # This means that for any prefix `P` ending at `node`, `node.best_candidate_info`
        # will store `(min_len, min_idx)` among all words `s` in wordsContainer
        # such that `s[::-1]` starts with `P`.
        for j, s in enumerate(wordsContainer):
            r_s = s[::-1]
            current_node = root
            
            # Update root's best_candidate_info. The root represents the empty prefix,
            # which corresponds to the empty common suffix. Every string shares an
            # empty suffix, so the root node must store the overall best candidate
            # (shortest length, then smallest index) from `wordsContainer`.
            current_len, current_idx = current_node.best_candidate_info
            if len(s) < current_len or (len(s) == current_len and j < current_idx):
                current_node.best_candidate_info = (len(s), j)

            for char in r_s:
                if char not in current_node.children:
                    current_node.children[char] = TrieNode()
                current_node = current_node.children[char]
                
                # Update the best_candidate_info for the current node.
                # This ensures that each node always stores the best candidate
                # for any string from wordsContainer whose reversed form passes
                # through this node.
                current_len, current_idx = current_node.best_candidate_info
                if len(s) < current_len or (len(s) == current_len and j < current_idx):
                    current_node.best_candidate_info = (len(s), j)
        
        ans = []
        # Process each query
        for q in wordsQuery:
            r_q = q[::-1]
            current_node = root
            
            # Initialize best candidate for this query.
            # Start with the best string that matches an empty suffix (from root).
            # This covers the case where no common suffix (LCSuf length > 0) is found.
            max_lcs_len = 0
            best_len_at_max_lcs, best_idx_at_max_lcs = root.best_candidate_info
            
            # Traverse the Trie with the reversed query string
            for i, char in enumerate(r_q):
                if char in current_node.children:
                    current_node = current_node.children[char]
                    current_lcs_len = i + 1 # Length of the common suffix found so far
                    
                    # Get the best candidate (length, original_index) associated
                    # with the current common prefix (which is a suffix in original strings).
                    potential_len, potential_idx = current_node.best_candidate_info
                    
                    # Apply the selection criteria:
                    # 1. Prioritize longer common suffix:
                    if current_lcs_len > max_lcs_len:
                        max_lcs_len = current_lcs_len
                        best_len_at_max_lcs = potential_len
                        best_idx_at_max_lcs = potential_idx
                    # 2. If common suffix length is the same, prioritize smaller string length:
                    elif current_lcs_len == max_lcs_len:
                        if potential_len < best_len_at_max_lcs:
                            best_len_at_max_lcs = potential_len
                            best_idx_at_max_lcs = potential_idx
                        # 3. If common suffix length and string length are the same,
                        #    the tie-breaking by original index is already handled:
                        #    `potential_idx` from `current_node.best_candidate_info` is
                        #    guaranteed to be the smallest index for its `potential_len`
                        #    and the current common suffix. So, no further action needed
                        #    if `potential_len == best_len_at_max_lcs`.
                else:
                    # No more common prefix (suffix in original strings) found.
                    # Break the loop, the current `best_idx_at_max_lcs` is the answer for this query.
                    break
            
            ans.append(best_idx_at_max_lcs)
            
        return ans