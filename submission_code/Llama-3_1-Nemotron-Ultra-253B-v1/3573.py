from collections import defaultdict
from bisect import bisect_right

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        target_counts = defaultdict(int)
        for c in word2:
            target_counts[c] += 1
        
        # Check if word1 has enough characters for each in target_counts
        for c in target_counts:
            if word1.count(c) < target_counts[c]:
                return 0
        
        L = len(word2)
        n = len(word1)
        if L == 0:
            return 0
        
        required_chars = list(target_counts.keys())
        
        # Precompute prefix sums for each required character
        prefix = {}
        for c in required_chars:
            prefix[c] = [0] * (n + 1)
            for i in range(n):
                prefix[c][i+1] = prefix[c][i] + (1 if word1[i] == c else 0)
        
        ans = 0
        
        for j in range(n):
            if j < L - 1:
                continue
            
            min_i = float('inf')
            for c in required_chars:
                current_total = prefix[c][j+1]
                required_value = current_total - target_counts[c]
                # Find the largest i where prefix[c][i] <= required_value
                i_c_max = bisect_right(prefix[c], required_value) - 1
                if i_c_max < 0:
                    min_i = -1
                    break
                min_i = min(min_i, i_c_max)
            
            if min_i == -1:
                continue
            
            upper_bound_i = min(min_i, j - L + 1)
            if upper_bound_i >= 0:
                ans += upper_bound_i + 1
        
        return ans