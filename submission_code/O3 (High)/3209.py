from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        """
        Dynamic programming.
        dp[i] – minimal number of coins required to obtain the first i fruits
                (i.e. fruits with indices 1 … i, 1-indexed).

        To compute dp[i] we consider the last fruit j that we actually buy
        (1 ≤ j ≤ i).  
        Buying fruit j costs prices[j] coins and gives us the fruits
        j, j+1, … , j+j for free.  
        Therefore, fruit i is covered by that purchase iff  j+j ≥ i.

        If we use j as the last purchase that covers i, all earlier fruits
        up to j-1 have already been paid for with optimal cost dp[j-1].
        Hence
              dp[i] = min(dp[j-1] + prices[j])  
                       for all j with 1 ≤ j ≤ i and j+j ≥ i
        The answer is dp[n] where n = len(prices).
        Time complexity: O(n²) (n ≤ 1000 → fast enough).
        Space complexity: O(n).
        """
        n = len(prices)
        # make prices 1-indexed
        prices = [0] + prices

        INF = 10**18
        dp = [INF]*(n+1)
        dp[0] = 0

        for i in range(1, n+1):
            # iterate j descending; once j+j < i further j will also fail
            for j in range(i, 0, -1):
                if j + j < i:
                    break
                dp[i] = min(dp[i], dp[j-1] + prices[j])

        return dp[n]