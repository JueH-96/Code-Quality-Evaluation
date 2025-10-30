class Solution:
    def maximizeTheProfit(self, n: int, offers: List[List[int]]) -> int:
        # Create a list of lists, where end_to_offers[e] holds all offers that end at house e
        end_to_offers = [[] for _ in range(n)]
        for s, e, gold in offers:
            end_to_offers[e].append((s, gold))
        
        # dp[i] will represent the maximum gold that can be obtained
        # by considering houses up to (and including) index i-1
        dp = [0] * (n + 1)
        
        for i in range(1, n + 1):
            # If we don't sell any range ending at house i-1,
            # the best we can do is whatever we had until house i-2
            dp[i] = dp[i - 1]
            
            # Check if there are offers ending at i-1
            for start_i, gold_i in end_to_offers[i - 1]:
                dp[i] = max(dp[i], dp[start_i] + gold_i)
        
        return dp[n]