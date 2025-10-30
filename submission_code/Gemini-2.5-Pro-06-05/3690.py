import math

class Solution:
  def minLength(self, s: str, numOps: int) -> int:
    n = len(s)

    def can_achieve(k: int) -> bool:
      """
      Check if it's possible to make the longest substring of identical characters
      have length at most k, using at most numOps flips.
      This is solved using dynamic programming with space optimization.
      """
      if k == 0:
        return False

      # dp[j][c]: min flips for the current prefix ending in a run of char c of length j
      # j is 1-indexed, from 1 to k.
      # c is 0 for '0', 1 for '1'.
      dp = [[math.inf] * 2 for _ in range(k + 1)]

      # Base case: prefix of length 1 (i=0)
      # Cost to make s[0] a '0'
      dp[1][0] = 1 if s[0] == '1' else 0
      # Cost to make s[0] a '1'
      dp[1][1] = 1 if s[0] == '0' else 0

      # Iterate through the rest of the string
      for i in range(1, n):
        # min_prev_0/1: min cost for prefix s[:i] to end in a run of 0/1 of any valid length (1 to k)
        min_prev_0 = math.inf
        min_prev_1 = math.inf
        for length in range(1, k + 1):
          min_prev_0 = min(min_prev_0, dp[length][0])
          min_prev_1 = min(min_prev_1, dp[length][1])
        
        new_dp = [[math.inf] * 2 for _ in range(k + 1)]

        # Cost to flip s[i] to '0' or '1'
        cost_to_0 = 1 if s[i] == '1' else 0
        cost_to_1 = 1 if s[i] == '0' else 0

        # --- Calculate new_dp for ending in '0' ---
        # Case 1: Start a new run of '0's (length 1).
        # This requires the previous character to be '1'.
        new_dp[1][0] = min_prev_1 + cost_to_0
        
        # Case 2: Extend an existing run of '0's.
        for j in range(2, k + 1):
          new_dp[j][0] = dp[j - 1][0] + cost_to_0
        
        # --- Calculate new_dp for ending in '1' ---
        # Case 1: Start a new run of '1's (length 1).
        # This requires the previous character to be '0'.
        new_dp[1][1] = min_prev_0 + cost_to_1
        
        # Case 2: Extend an existing run of '1's.
        for j in range(2, k + 1):
          new_dp[j][1] = dp[j - 1][1] + cost_to_1
        
        dp = new_dp

      # After iterating through the whole string, find the minimum flips required.
      min_flips = math.inf
      for j in range(1, k + 1):
        min_flips = min(min_flips, dp[j][0], dp[j][1])
      
      return min_flips <= numOps

    # Binary search for the minimum possible length k.
    low, high = 1, n
    ans = n
    while low <= high:
      mid = (low + high) // 2
      if can_achieve(mid):
        ans = mid
        high = mid - 1
      else:
        low = mid + 1
        
    return ans