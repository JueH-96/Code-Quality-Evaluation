import collections

class Solution:
  def maxTotalReward(self, rewardValues: list[int]) -> int:
    # Sort and get unique reward values based on the reasoning that
    # picked rewards must be in increasing order of value, and
    # at most one reward of any specific value can be picked.
    unique_rewards = sorted(list(set(rewardValues)))
    
    # Maximum reward value is 2000. Max sum is < 2 * 2000 = 4000.
    # dp[k] is true if sum k is achievable. Indices 0 to 3999.
    MAX_SUM_LIMIT = 4000 
    
    dp = [False] * MAX_SUM_LIMIT
    dp[0] = True # Base case: sum 0 is achievable.
    
    max_achieved_sum = 0

    for r_val in unique_rewards:
      # For each reward r_val, iterate through existing achievable sums `x`.
      # We are interested in sums `x` such that `dp[x]` is true AND `x < r_val`.
      # If so, we can pick r_val, and `x + r_val` becomes a new achievable sum.
      # Iterate x downwards to ensure dp[x] is based on sums from *previous* rewards.
      for x in range(r_val - 1, -1, -1):
        if dp[x]:
          new_sum = x + r_val
          
          # new_sum is guaranteed to be < MAX_SUM_LIMIT because:
          # max x here is r_val - 1. So max x is 2000 - 1 = 1999 (if r_val is 2000).
          # max new_sum = (r_val - 1) + r_val = 2 * r_val - 1.
          # If r_val <= 2000, then max new_sum is 2 * 2000 - 1 = 3999.
          # This fits into an array of size 4000 (indices 0-3999).
          # So, the check `if new_sum < MAX_SUM_LIMIT:` is technically not needed here
          # given the problem constraints, but good for robustness if constraints changed.
          
          # If dp[new_sum] is already true, we don't need to set it again,
          # but max_achieved_sum still needs to be updated if new_sum is larger.
          # Not checking `if not dp[new_sum]` before `dp[new_sum] = True` is fine.
          dp[new_sum] = True
          max_achieved_sum = max(max_achieved_sum, new_sum)
            
    return max_achieved_sum