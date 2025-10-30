from typing import List
from collections import defaultdict

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        if not words:
            return 0 if not target else -1
        
        base = 911382629
        mod = 10**18 + 3
        
        # Preprocess words into a dictionary: length -> {hash: min_cost}
        word_map = defaultdict(dict)
        max_len = 0
        for word, cost in zip(words, costs):
            l = len(word)
            max_len = max(max_len, l)
            # Compute the hash of the word
            h = 0
            for c in word:
                h = (h * base + ord(c)) % mod
            # Update the word_map for this length and hash
            if h not in word_map[l] or cost < word_map[l][h]:
                word_map[l][h] = cost
        
        n = len(target)
        if n == 0:
            return 0
        
        # Compute prefix_hash for the target
        prefix_hash = [0] * (n + 1)
        for i in range(n):
            prefix_hash[i+1] = (prefix_hash[i] * base + ord(target[i])) % mod
        
        # Precompute powers of the base up to the maximum word length
        power = [1] * (max_len + 1)
        for i in range(1, max_len + 1):
            power[i] = (power[i-1] * base) % mod
        
        # Initialize DP array
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        # Fill DP array
        for i in range(1, n + 1):
            for l in word_map:
                if i < l:
                    continue
                # Compute the hash of the substring target[i-l:i]
                current_hash = (prefix_hash[i] - prefix_hash[i - l] * power[l]) % mod
                if current_hash < 0:
                    current_hash += mod
                # Check if this hash exists in the word_map for length l
                if current_hash in word_map[l]:
                    candidate_cost = dp[i - l] + word_map[l][current_hash]
                    if candidate_cost < dp[i]:
                        dp[i] = candidate_cost
        
        return dp[n] if dp[n] != float('inf') else -1