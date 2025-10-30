from typing import List

class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        # Build the trie with all prefixes from all words
        root = {}
        for word in words:
            node = root
            prefix = ""
            for char in word:
                prefix += char
                if prefix not in node:
                    node[prefix] = {}
                node = node[prefix]
                # Mark the end of each prefix
                node['$'] = True
        
        target_length = len(target)
        dp = [float('inf')] * (target_length + 1)
        dp[0] = 0  # base case: empty string needs 0 prefixes
        
        for i in range(target_length):
            if dp[i] == float('inf'):
                continue
            node = root
            j = i
            match_length = 0
            while j < target_length:
                char = target[j]
                prefix = target[i:j+1]
                if prefix not in node:
                    break
                node = node[prefix]
                if '$' in node:
                    match_length = j - i + 1
                    dp[j + 1] = min(dp[j + 1], dp[i] + 1)
                j += 1
            if match_length == 0:
                continue
        
        return dp[target_length] if dp[target_length] != float('inf') else -1