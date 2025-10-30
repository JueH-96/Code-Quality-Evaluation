from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # Compute the "score array" where a clear (1) -> +1, fail (0) -> -1
        # Then prefix_sum[k] = sum of scores for levels [0..k-1]
        total = 0
        for p in possible:
            total += (1 if p == 1 else -1)
        
        prefix = 0
        # We need to find the smallest k in [1..n-1] such that:
        #    prefix_sum[k] > total - prefix_sum[k]
        # => 2 * prefix_sum[k] > total
        for k in range(1, n):
            prefix += (1 if possible[k-1] == 1 else -1)
            if 2 * prefix > total:
                return k
        
        return -1