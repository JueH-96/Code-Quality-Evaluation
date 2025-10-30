from collections import Counter
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        if not power:
            return 0
        freq = Counter(power)
        distinct = sorted(freq.keys())
        n = len(distinct)
        T = [d * freq[d] for d in distinct]
        
        if n == 0:
            return 0
        
        dp = [0] * n
        running_max = 0
        j = 0
        best_so_far = 0
        
        for i in range(n):
            while j < i and distinct[j] <= distinct[i] - 3:
                if dp[j] > best_so_far:
                    best_so_far = dp[j]
                j += 1
            
            candidate = T[i] + best_so_far
            if i == 0:
                dp[i] = candidate
            else:
                dp[i] = max(running_max, candidate)
            
            if dp[i] > running_max:
                running_max = dp[i]
                
        return running_max