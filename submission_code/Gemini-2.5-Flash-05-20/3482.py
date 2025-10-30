import collections
from typing import List

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        N = len(target)
        
        # Step 1: Build the Trie for efficient word lookup
        # Each node in the trie is a dictionary.
        # A special key '_cost' stores the minimum cost of a word ending at that node.
        trie = {}
        for j in range(len(words)):
            word = words[j]
            cost = costs[j]
            
            curr_node = trie
            for char in word:
                if char not in curr_node:
                    curr_node[char] = {}
                curr_node = curr_node[char]
            
            # Store the minimum cost for the word ending at this node.
            # This handles cases where identical words might appear with different costs.
            if '_cost' not in curr_node:
                curr_node['_cost'] = cost
            else:
                curr_node['_cost'] = min(curr_node['_cost'], cost)
        
        # Step 2: Dynamic Programming
        # dp[i] will store the minimum cost to form the prefix target[:i]
        # Initialize with infinity, except for dp[0] which is 0.
        dp = [float('inf')] * (N + 1)
        dp[0] = 0  # Cost to form an empty string is 0
        
        # Iterate through each possible starting position `i` in the target string
        for i in range(N):
            # If target[:i] cannot be formed, then we cannot extend from this point.
            if dp[i] == float('inf'):
                continue
            
            # Traverse the Trie starting from the character `target[i]`
            curr_trie_node = trie
            
            # `k` represents the current character index in `target` being matched,
            # effectively building a substring `target[i : k+1]`.
            for k in range(i, N):
                char = target[k]
                
                # If the current character `char` does not lead to a valid path in the Trie
                # from `curr_trie_node`, then no word starting at `target[i]` can be
                # extended further to include `target[k]`. Break and move to next `i`.
                if char not in curr_trie_node:
                    break
                
                curr_trie_node = curr_trie_node[char]
                
                # If `curr_trie_node` marks the end of a word (i.e., has a '_cost'),
                # it means `target[i : k+1]` forms a valid word.
                if '_cost' in curr_trie_node:
                    word_cost = curr_trie_node['_cost']
                    
                    # The length of this matched word is `k - i + 1`.
                    # `target[i : k+1]` corresponds to an `(i+word_len)` position in dp array.
                    # We update `dp[k+1]` (cost to form prefix up to `k+1`)
                    # with `dp[i]` (cost to form prefix up to `i`) plus the `word_cost`.
                    dp[k + 1] = min(dp[k + 1], dp[i] + word_cost)
                    
        # Step 3: Return result
        # If dp[N] is still infinity, it means the target string cannot be formed.
        if dp[N] == float('inf'):
            return -1
        else:
            return dp[N]