from typing import List
import math

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)

        # dp[i] will store the minimum cost to acquire fruits from index 0 to i.
        # The array size is n, covering indices 0 to n-1.
        dp = [float('inf')] * n

        # Base case: To acquire fruit at index 0, we must purchase it.
        dp[0] = prices[0]

        # Iterate through each fruit index `i` from 0 to n-1.
        # Consider the strategy where we purchase fruit `i`.
        # The cost incurred by this specific strategy is the minimum cost
        # to acquire fruits 0 to i-1, plus the cost of purchasing fruit i.
        # The minimum cost to acquire fruits 0 to i-1 is stored in dp[i-1] (if i > 0).
        # If i == 0, the cost to acquire 0 fruits is 0.
        for i in range(n):
            # `cost_to_acquire_up_to_i_minus_1` represents the minimum cost to acquire
            # fruits from index 0 up to index i-1.
            # For i = 0, this is the cost to acquire 0 fruits, which is 0.
            # For i > 0, this is dp[i-1].
            # We use `if i > 0 else 0` to handle the i=0 case where dp[i-1] is not defined.
            # This calculates the cost of the path that acquires 0..i-1 optimally AND THEN buys fruit i.
            cost_to_acquire_up_to_i_minus_1 = dp[i-1] if i > 0 else 0
            cost_if_we_buy_i = cost_to_acquire_up_to_i_minus_1 + prices[i]

            # This purchase of fruit i covers a range of indices starting from i.
            # According to the rule (using 0-indexing), purchasing fruit at index `i`
            # (which is the (i+1)-th fruit) gives the next `i+1` fruits for free.
            # These free fruits are at indices i+1, i+2, ..., i + (i+1) = 2*i + 1.
            # So, the purchase of fruit i covers indices from i (paid) up to 2*i + 1 (free).
            # The range of indices `k` covered is [i, 2*i + 1].
            # For every index `k` within this covered range [i, 2*i+1] (and within bounds [0, n-1]),
            # the cost `cost_if_we_buy_i` is a potential candidate for the minimum cost
            # to acquire fruits from index 0 up to index `k` (dp[k]).
            
            # The inclusive end index for k is min(n-1, 2*i + 1).
            end_k = min(n - 1, 2 * i + 1)
            
            # Iterate through all indices `k` that are covered by purchasing fruit `i`.
            # Update dp[k] if `cost_if_we_buy_i` is less than the current minimum cost for dp[k].
            for k in range(i, end_k + 1):
                 # Update the minimum cost to acquire fruits 0..k
                dp[k] = min(dp[k], cost_if_we_buy_i)

        # After iterating through all possible purchase options (from fruit 0 to n-1),
        # dp[n-1] will contain the minimum cost to acquire fruits from index 0 to n-1, which is all fruits.
        return dp[n-1]