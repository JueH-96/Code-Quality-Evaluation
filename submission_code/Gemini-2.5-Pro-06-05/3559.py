from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        
        # 1. Build a Trie from the words.
        # Each node is a dictionary mapping a character to the next node.
        # Any path from the root represents a valid prefix.
        trie_root = {}
        for word in words:
            node = trie_root
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
        
        # 2. DP initialization
        n = len(target)
        # dp[i] will store the minimum number of valid strings to form target[:i].
        dp = [float('inf')] * (n + 1)
        # Base case: 0 strings are needed for an empty prefix.
        dp[0] = 0
        
        # 3. DP calculation
        # Iterate through all possible start points `i` of a new valid string.
        for i in range(n):
            # If target[:i] is unreachable, we can't extend from this point.
            if dp[i] == float('inf'):
                continue
            
            # Start a Trie traversal from the root for the substring target[i:].
            node = trie_root
            # `k` is the end index of the potential valid prefix `target[i:k+1]`.
            for k in range(i, n):
                char = target[k]
                
                # If the character path does not exist in the Trie, then `target[i:k+1]`
                # is not a valid prefix. Any longer string starting at `i` will also
                # be invalid, so we can break out of this inner loop.
                if char not in node:
                    break
                
                # Move to the next node in the Trie.
                node = node[char]
                
                # Since a path exists, `target[i:k+1]` is a valid prefix.
                # Update the DP state for the end of this prefix.
                # The cost is `dp[i]` (to form target[:i]) + 1 (for target[i:k+1]).
                dp[k + 1] = min(dp[k + 1], dp[i] + 1)
                
        # 4. Result
        result = dp[n]
        
        # If result is still infinity, the target cannot be formed.
        if result == float('inf'):
            return -1
        else:
            return int(result)