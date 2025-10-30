from typing import List
import collections

class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Group offers by their end index
        offers_by_end = collections.defaultdict(list)
        for start, end, gold in offers:
            offers_by_end[end].append((start, gold))
        
        # dp[i] will hold the max profit obtainable from houses [0..i]
        dp = [0] * n
        
        # Process each house index in increasing order
        for i in range(n):
            # By default, carry over the previous maximum
            if i > 0:
                dp[i] = dp[i - 1]
            
            # Consider any offer that ends at i
            for start, gold in offers_by_end.get(i, []):
                prev_profit = dp[start - 1] if start > 0 else 0
                dp[i] = max(dp[i], prev_profit + gold)
        
        # The answer is the max profit using houses [0..n-1]
        return dp[n - 1]