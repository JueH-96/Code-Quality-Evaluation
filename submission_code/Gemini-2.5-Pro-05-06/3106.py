from typing import List

class Solution:
  def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
    # dp[j] will store the maximum length of a subsequence that sums to j.
    # Initialize dp[0] = 0: an empty subsequence sums to 0 with length 0.
    # Initialize dp[j] = -1 for j > 0: initially, sum j is not achievable.
    # -1 acts as a sentinel; actual lengths are non-negative (>= 0).
    
    dp = [-1] * (target + 1)
    dp[0] = 0
    
    # Process each number from the input list nums
    for num in nums:
      # Iterate j from target down to num.
      # This descending order for j is crucial for 0/1 knapsack-type problems.
      # It ensures that each 'num' from the input 'nums' is considered at most once
      # when forming the sum 'j' in the current pass for 'num'.
      # If we iterated j upwards, dp[j - num] might have already been updated
      # using the *current* 'num', effectively allowing 'num' to be used multiple times
      # to make up sum 'j'. This is not the standard subsequence behavior unless
      # 'nums' itself has duplicates (which are correctly handled by iterating
      # through each element of 'nums' in the outer loop).
      for j in range(target, num - 1, -1):
        # Check if the sum (j - num) is achievable (i.e., dp[j - num] is not -1)
        if dp[j - num] != -1:
          # If sum (j - num) is achievable with length L = dp[j - num],
          # then sum j can be achieved by including the current 'num'.
          # The length of such a subsequence would be L + 1.
          
          # We update dp[j] if this new way of forming sum 'j' (by including 'num')
          # results in a longer subsequence.
          # The max function handles all cases:
          # - If dp[j] is -1 (sum j was unachievable), it becomes dp[j - num] + 1.
          # - If dp[j] was already a positive length, it's updated if dp[j - num] + 1
          #   is greater, thus maximizing the length.
          dp[j] = max(dp[j], dp[j - num] + 1)
            
    # After processing all numbers, dp[target] holds the maximum length
    # of a subsequence summing to target.
    # If dp[target] is still -1, it means the target sum was not achievable.
    return dp[target]