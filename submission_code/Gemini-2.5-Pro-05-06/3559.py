from typing import List

# Definition for TrieNode to be used by the Solution class
class TrieNode:
    def __init__(self):
        self.children = {}
        # No specific flag like is_end_of_word or is_prefix is needed here.
        # Any path from the root of the Trie corresponds to a prefix of some word
        # in the input `words`, and thus represents a "valid string".

class Solution:
  def minValidStrings(self, words: List[str], target: str) -> int:
    trie_root = TrieNode()

    # Build the Trie from all prefixes of words in `words`.
    # For each word, iterate through its characters and add them to the Trie.
    # Each node in the path represents a prefix.
    for word in words:
      node = trie_root
      for char in word:
        if char not in node.children:
          node.children[char] = TrieNode()
        node = node.children[char]
        
    n = len(target)
    
    # dp[k] will store the minimum number of valid strings 
    # required to form the prefix target[0...k-1] of length k.
    # The size of the dp array is n + 1.
    # dp[0] corresponds to an empty prefix, requiring 0 strings.
    # Initialize all other dp values to infinity.
    dp = [float('inf')] * (n + 1)
    dp[0] = 0 
    
    # Iterate through the target string to fill the dp table.
    # dp[i] represents the minimum strings for target[0...i-1].
    # We consider i as the starting point of a new segment target[i...].
    for i in range(n): 
      # If dp[i] is infinity, it means target[0...i-1] cannot be formed.
      # So, we cannot extend from this state.
      if dp[i] == float('inf'):
        continue
      
      current_trie_node = trie_root # Start matching a new segment from the root of the Trie.
      
      # Try to form a segment target[i...j] (inclusive).
      # This segment's characters are target[i], target[i+1], ..., target[j].
      for j in range(i, n):
        char = target[j] # Current character to match in the Trie.
        
        if char not in current_trie_node.children:
          # If the current character `char` is not a child of `current_trie_node`,
          # then the string target[i...j] is not a prefix of any word in `words`
          # (when trying to match target[i...j] character by character in the Trie).
          # All further extensions (e.g., target[i...j+1]) also won't be.
          # So, break from this inner loop for j.
          break 
        
        # Move to the next node in the Trie.
        current_trie_node = current_trie_node.children[char]
        
        # At this point, the string target[i...j] has been successfully traversed
        # in the Trie starting from trie_root. This means target[i...j] is a
        # prefix of some word in `words`, and is therefore a "valid string".
        
        # We can form the prefix target[0...j] by:
        # 1. Forming target[0...i-1] (which costs dp[i] strings).
        # 2. Appending the current valid string target[i...j] (which costs 1 string).
        # The total number of strings is dp[i] + 1.
        # We update dp[j+1] (cost for target[0...j]) if this path is better.
        dp[j+1] = min(dp[j+1], dp[i] + 1)
          
    # After filling the dp table, dp[n] contains the minimum number of
    # valid strings to form the entire target string target[0...n-1].
    final_ans = dp[n]
    
    # If dp[n] is still infinity, it means the target string cannot be formed.
    if final_ans == float('inf'):
      return -1
    else:
      # Finite values in dp will be integers as they are sums of 1s.
      return int(final_ans)