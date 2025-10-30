import math

class Solution:
  def minimumCoins(self, prices: list[int]) -> int:
    n = len(prices)
    
    # dp[k] will store the minimum cost to acquire the first k fruits (1-indexed count).
    # So dp[k] means fruits with 0-indexed indices 0, 1, ..., k-1 are acquired.
    # The dp array is of size n+1.
    # dp[0] is the base case: 0 cost to acquire 0 fruits.
    # dp[1] will be the minimum cost to acquire fruit 0 (the first fruit).
    # ...
    # dp[n] will be the minimum cost to acquire all fruits (indices 0 to n-1).
    dp = [math.inf] * (n + 1)
    dp[0] = 0
    
    # Iterate through each fruit `p_idx` (0-indexed) considering it as the fruit to purchase.
    # `p_idx` corresponds to the (p_idx + 1)-th fruit in 1-indexed terms.
    for p_idx in range(n):
      # To be able to purchase fruit `p_idx`, we must have already acquired fruits 0, ..., p_idx-1.
      # The minimum cost for this state (first p_idx fruits acquired) is dp[p_idx].
      # If dp[p_idx] is math.inf, it implies this state is currently unreachable.
      # However, in this problem's structure where dp[0]=0, dp[p_idx] will be finite
      # if p_idx is reachable by previous purchases.
      if dp[p_idx] == math.inf:
          continue # Should not happen with p_idx in range(n) given dp[0]=0

      # Cost if we decide to purchase fruit `p_idx`:
      # It's the cost to acquire fruits 0..p_idx-1 (which is dp[p_idx])
      # plus the cost of fruit `p_idx` itself (which is prices[p_idx]).
      current_total_cost = dp[p_idx] + prices[p_idx]
      
      # If we purchase the (p_idx + 1)-th fruit (0-indexed `p_idx`):
      # We get the next (p_idx + 1) fruits for free.
      # In 0-indexed terms:
      # - Fruit `p_idx` is purchased.
      # - Fruits `p_idx + 1`, `p_idx + 2`, ..., `p_idx + (p_idx + 1)` are free.
      # The last 0-indexed fruit that is covered by this purchase action (either `p_idx` itself or one of the free ones)
      # is `p_idx + (p_idx + 1) = 2*p_idx + 1`.
      # This index must be capped by `n-1` (the actual last fruit index in the market).
      last_actual_fruit_idx_covered = min(n - 1, 2 * p_idx + 1)
      
      # This `current_total_cost` is a candidate cost for acquiring all fruits
      # from index 0 up to `kth_fruit_idx_covered`.
      # The dp state `dp[j]` means fruits `0...j-1` are covered (i.e., first `j` fruits).
      # So, if fruits `0...kth_fruit_idx_covered` are covered, this means `kth_fruit_idx_covered + 1` fruits are covered.
      # This value `current_total_cost` becomes a candidate for `dp[kth_fruit_idx_covered + 1]`.
      
      # We iterate through all fruits that this purchase covers (from `p_idx` up to `last_actual_fruit_idx_covered`).
      # For each such `kth_fruit_idx_covered`, we update `dp[kth_fruit_idx_covered + 1]`.
      for kth_fruit_idx_covered in range(p_idx, last_actual_fruit_idx_covered + 1):
        dp_update_idx = kth_fruit_idx_covered + 1 # Target index in dp array (count of fruits)
        dp[dp_update_idx] = min(dp[dp_update_idx], current_total_cost)
             
    # The final answer is the minimum cost to acquire all n fruits (0 to n-1),
    # which is stored in dp[n].
    return dp[n]