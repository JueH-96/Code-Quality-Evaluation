from typing import List

class Solution:
  def minimumCoins(self, prices: List[int]) -> int:
    """
    Calculates the minimum number of coins needed to acquire all fruits.

    This problem is solved using dynamic programming. Let dp[i] be the minimum
    cost to acquire fruits from 1 to i. The goal is to find dp[n].

    The recurrence relation is derived by considering the last purchase made to
    acquire fruit i. If we buy fruit j (1 <= j <= i), this purchase covers
    fruits up to 2j. For this to cover fruit i, we need i <= 2j.
    Thus, for a given i, we look for the optimal j in [ceil(i/2), i]
    that minimizes the total cost, which is (cost to get up to j-1) + (price of j).
    dp[i] = min_{ceil(i/2) <= j <= i} (dp[j-1] + prices[j-1])
    """
    n = len(prices)
    
    # dp[i] stores the minimum cost to acquire fruits 1...i.
    # dp is 1-indexed for fruits, so its size is n+1.
    # dp[0] = 0 (base case for 0 fruits).
    dp = [0] * (n + 1)
    
    for i in range(1, n + 1):
        # To acquire fruit i, we must have made a final purchase of fruit j
        # such that this purchase covers up to fruit i.
        # Buying fruit j (cost prices[j-1]) makes fruits j+1...2j free.
        # This covers i if j <= i <= 2j, which means ceil(i/2) <= j <= i.
        # The cost for such a path is dp[j-1] + prices[j-1].
        # We minimize this over all valid j.
        
        # In 1-based indexing, j_start is ceil(i/2).
        # Using integer arithmetic, this is (i + 1) // 2.
        j_start = (i + 1) // 2
        
        min_cost = float('inf')
        for j in range(j_start, i + 1):
            cost = dp[j-1] + prices[j-1]
            if cost < min_cost:
                min_cost = cost
        
        dp[i] = min_cost
        
    return dp[n]