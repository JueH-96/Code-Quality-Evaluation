class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        from collections import defaultdict
        
        # group offers by their ending house
        ends_map = defaultdict(list)
        for start, end, gold in offers:
            ends_map[end].append((start, gold))
        
        # dp[i] = max profit from selling houses up to (but not including) house i
        dp = [0] * (n + 1)
        
        for i in range(n):
            # carry forward the previous profit if we skip house i
            dp[i+1] = dp[i]
            
            # for every offer that ends at house i, check if taking it increases profit
            if i in ends_map:
                for start, gold in ends_map[i]:
                    dp[i+1] = max(dp[i+1], dp[start] + gold)
        
        return dp[n]