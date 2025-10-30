from collections import Counter
from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        freq = Counter(power)
        unique_x = sorted(freq.keys())
        n = len(unique_x)
        
        if n == 0:
            return 0
        elif n == 1:
            return unique_x[0] * freq[unique_x[0]]
        
        dp = [0] * (n)
        dp[0] = unique_x[0] * freq[unique_x[0]]
        
        # Handle the second element
        if unique_x[1] > unique_x[0] + 2:
            dp[1] = dp[0] + unique_x[1] * freq[unique_x[1]]
        else:
            dp[1] = max(dp[0], unique_x[1] * freq[unique_x[1]])
        
        for i in range(2, n):
            x = unique_x[i]
            current = x * freq[x]
            
            if x > unique_x[i-1] + 2:
                dp[i] = dp[i-1] + current
            else:
                option1 = dp[i-1]
                option2 = dp[i-2] + current
                if x > unique_x[i-2] + 2:
                    option2 = max(option2, dp[i-2] + current)
                dp[i] = max(option1, option2)
        
        return dp[-1]