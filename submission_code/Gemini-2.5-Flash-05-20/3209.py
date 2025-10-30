import math

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        
        # dp[i] will store the minimum cost to acquire all fruits from index 0 up to i-1.
        # Our goal is dp[n], which represents acquiring all fruits from index 0 to n-1.
        dp = [math.inf] * (n + 1)
        
        # Base case: The cost to acquire 0 fruits is 0.
        dp[0] = 0
        
        # Iterate through each fruit, considering it as a potential purchase point.
        # `i` is the 0-indexed position of the fruit currently being considered for purchase.
        for i in range(n):
            # If dp[i] is infinity, it means we cannot reach the state where
            # fruits 0 to i-1 are acquired with a finite cost.
            # Thus, we cannot make a purchase starting from fruit `i` through this path.
            if dp[i] == math.inf:
                continue
            
            # Calculate the total cost if we purchase fruit `i`.
            # This cost is the minimum cost to acquire fruits up to i-1 (dp[i])
            # plus the price of the current fruit (prices[i]).
            current_total_cost = dp[i] + prices[i]
            
            # According to the offer:
            # If we purchase the (i+1)-th fruit (at 0-indexed position `i`),
            # we get the next (i+1) fruits for free.
            # This means fruits from `i+1` up to `i + (i+1)` (which is `2*i + 1`) are free.
            # So, the purchase effectively covers all fruits from index `i` up to `2*i + 1`.
            
            # We need to update dp states for all indices `j` where fruits up to `j-1`
            # are now acquired due to this purchase.
            # These `j` indices range from `i + 1` (meaning fruit `i` is acquired)
            # up to `2*i + 2` (meaning fruits up to `2*i + 1` are acquired).
            # The `min(n + 1, ...)` ensures we don't go out of bounds of the dp array.
            
            # The loop goes from `i + 1` because `dp[i]` already accounts for fruits up to `i-1`.
            # `dp[i+1]` accounts for fruits up to `i` (meaning fruit `i` is the last one covered).
            # `dp[min(n+1, 2*i + 2)]` means fruits up to `min(n, 2*i+1)` are covered.
            for j in range(i + 1, min(n + 1, 2 * i + 2)):
                dp[j] = min(dp[j], current_total_cost)
                
        # The final answer is dp[n], which is the minimum cost to acquire all fruits
        # from index 0 to n-1.
        return dp[n]