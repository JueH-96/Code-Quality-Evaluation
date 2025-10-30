from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # Convert possible array to score changes: +1 for 1, -1 for 0
        scores = [1 if x == 1 else -1 for x in possible]
        
        # Compute total sum of scores (what Bob would get if Alice takes 0 levels)
        total_sum = sum(scores)
        
        # Now compute prefix sum for Alice and check condition:
        # Alice takes k levels => her score = prefix_sum[k]
        # Bob takes the rest => his score = total_sum - prefix_sum[k]
        # We want prefix_sum[k] > total_sum - prefix_sum[k]
        # => 2 * prefix_sum[k] > total_sum
        prefix = 0
        # k must be at least 1 and at most n-1 (both play >=1 level)
        for k in range(1, n):
            prefix += scores[k-1]
            if 2 * prefix > total_sum:
                return k
        
        return -1