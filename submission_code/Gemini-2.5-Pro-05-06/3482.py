import math
from typing import List

class Solution:
  def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
    n = len(target)
    
    # Trie node structure:
    # A dictionary where keys are characters.
    # The value for a character key is another dictionary representing the child node.
    # A special key, e.g., '_cost_', stores the cost if the path up to this node forms a word.
    trie_root = {}

    # Populate the Trie with words and their minimum costs
    # Total character processing for trie construction is sum of lengths of words in input `words` array.
    # This is S_L_total, which is <= 5*10^4 according to constraints.
    for i in range(len(words)):
        word = words[i]
        cost = costs[i]
        
        node = trie_root
        for char_in_word in word:
            if char_in_word not in node:
                node[char_in_word] = {} # Create child node (empty dict initially)
            node = node[char_in_word] # Move to child node
        
        # Mark end of word and store minimum cost for this specific word string
        current_min_cost_for_this_word_str = node.get('_cost_', float('inf'))
        node['_cost_'] = min(current_min_cost_for_this_word_str, cost)

    # Dynamic Programming
    # dp[k] = minimum cost to form target[:k] (prefix of target of length k)
    # dp array size is n+1. Indices 0 to n.
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  # Cost to form an empty string (target[:0]) is 0

    # Iterate i from 0 to n-1.
    # dp[i] is the known minimum cost to form prefix target[:i].
    # We try to append words from `words` that match target[i...j].
    for i in range(n): 
        if dp[i] == float('inf'):
            # Cannot form target[:i], so cannot extend from it.
            continue

        # current_trie_path_node traces the path in the Trie corresponding to target[i...j]
        current_trie_path_node = trie_root
        
        # j iterates from i to n-1.
        # char_in_target is target[j].
        # The substring being matched against the Trie is target[i...j].
        for j in range(i, n): 
            char_in_target = target[j]
            
            if char_in_target not in current_trie_path_node:
                # target[i...char_in_target] (i.e., target[i...j]) cannot be extended further
                # to match any word prefix in Trie.
                # This means no word in `words` starts with target[i...j].
                break 
            
            # Advance in the Trie
            current_trie_path_node = current_trie_path_node[char_in_target]
            
            # Check if the path target[i...j] corresponds to a word in the Trie
            if '_cost_' in current_trie_path_node:
                # target[i...j] is a word.
                word_cost = current_trie_path_node['_cost_']
                
                # Length of the matched word target[i...j] is (j - i + 1).
                word_len = (j - i + 1)
                
                # If we append this word to target[:i] (cost dp[i]),
                # we form target[:i+word_len] with cost dp[i] + word_cost.
                # Update dp[i + word_len] if this path is cheaper.
                dp_idx_to_update = i + word_len
                
                if dp[i] + word_cost < dp[dp_idx_to_update]:
                    dp[dp_idx_to_update] = dp[i] + word_cost
    
    result = dp[n] # Cost to form the full target string target[:n]
    
    return int(result) if result != float('inf') else -1