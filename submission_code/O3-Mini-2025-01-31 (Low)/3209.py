from typing import List
import math

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        # We reinterpret the problem as follows.
        # We have fruits numbered 1..n (the input array is 1-indexed).
        # When you purchase the k‑th fruit (cost = prices[k‑1]), you receive that fruit plus a “free‐offer”
        # that lets you take the next k fruits (if they exist) for free.
        # In other words, the purchase at position k “covers” the fruits from k to min(n, k + k).
        #
        # Now, you must select a set S ⊆ {1,...,n} of positions at which you purchase fruits so that every fruit 
        # in 1..n is “covered” by at least one purchase. (A purchase at position j covers fruits j, j+1, …, min(n, j+j)).
        # Your goal is to minimize the total cost = sum(prices[j‑1] for j in S).
        #
        # One way to solve such a covering problem is to use dynamic programming.
        # Let dp[i] represent the minimum cost needed to cover fruits i, i+1, …, n.
        # Then, to cover fruit i, we must “activate” (i.e. purchase) a fruit at some position j (with j between ceil(i/2) and i)
        # that is able to cover fruit i. Why? Because a purchase made at j covers fruits j up to j+j.
        # The condition for fruit i to be covered by an offer from j is:
        #    j ≤ i ≤ j+j.
        #
        # So for fixed i we consider all valid j from ceil(i/2) to i.
        # If we purchase fruit j (cost = prices[j‑1]), its offer covers up to fruit (j+j).
        # Then the next fruit we would need to cover is fruit (j+j+1).
        #
        # Therefore, the recurrence is:
        #    dp[i] = min { prices[j‑1] + dp[j+j+1] }  for j from ceil(i/2) to i.
        # We set dp[n+1] = 0 (or more generally, dp[k] = 0 for k > n).
        #
        # Finally, the answer is dp[1] – the minimum cost to cover all fruits 1..n.
        #
        # Time complexity: O(n^2) which is fine for n up to 1000.
        
        n = len(prices)
        # We use a 1-indexed DP array.
        dp = [0] * (n + 2)  # dp[i] for i from 1 to n; dp[n+1] = 0
        
        # Fill dp from n down to 1.
        for i in range(n, 0, -1):
            best = float('inf')
            # For fruit i to be covered by a purchase at j,
            # we need j ≤ i ≤ j+j, i.e. j must be at least ceil(i/2) and at most i.
            # Using integer arithmetic, ceil(i/2) can be computed as (i+1)//2.
            start = (i + 1) // 2
            for j in range(start, i + 1):
                if j > n:
                    continue
                # When we purchase fruit j, the offer covers fruits j through min(n, j+j).
                # Therefore, after this offer, the next fruit index to cover is j+j+1.
                next_index = j + j + 1
                if next_index > n + 1:
                    next_index = n + 1
                cost = prices[j - 1] + dp[next_index]
                best = min(best, cost)
            dp[i] = best
        return dp[1]


# Sample testing:
if __name__ == '__main__':
    sol = Solution()
    # Example 1
    prices = [3, 1, 2]
    print(sol.minimumCoins(prices))  # Expected output: 4
    
    # Example 2
    prices = [1, 10, 1, 1]
    print(sol.minimumCoins(prices))  # Expected output: 2