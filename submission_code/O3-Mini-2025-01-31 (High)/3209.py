from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        # dp[i] will be the minimum cost to acquire fruits 1 through i (fruits are numbered 1-indexed)
        # Let dp[0] = 0 (no fruits needed, no cost)
        dp = [float("inf")] * (n + 1)
        dp[0] = 0

        # Explanation:
        # When you purchase the j-th fruit (1-indexed) for prices[j-1] coins,
        # you get the j-th fruit plus the next j fruits free. That means, if this
        # is the last purchase in our “chain”, then fruits numbered j through 2*j are acquired.
        # Hence, if j is the last fruit bought and the chain covers fruits 1..i, we must have:
        #      i <= 2 * j   (or equivalently, j >= ceil(i/2) )
        # And the cost of that chain is: dp[j-1] (cost to cover fruits 1 .. (j-1))
        # plus the cost for purchasing fruit j: prices[j-1].
        # So for each i from 1 to n we can write:
        #      dp[i] = min { dp[j-1] + prices[j-1]  for j from ceil(i/2) to i }.
        #
        # (Note that even if a fruit could be taken for free from a previous purchase,
        # we might want to "upgrade" it by buying it so that its free offer covers more fruits.)
        
        for i in range(1, n + 1):
            # The last purchased fruit that covers fruit i must be at least ceil(i/2).
            j_min = (i + 1) // 2  
            for j in range(j_min, i + 1):
                dp[i] = min(dp[i], dp[j - 1] + prices[j - 1])
        return dp[n]