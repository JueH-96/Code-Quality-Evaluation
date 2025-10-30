import math
from typing import List

class Solution:
  """
  Finds the length of the longest subsequence of nums such that the absolute differences 
  between consecutive elements form a non-increasing sequence.
  """
  def longestSubsequence(self, nums: List[int]) -> int:
    """
    Args:
      nums: A list of integers. Constraints: 2 <= nums.length <= 10^4, 1 <= nums[i] <= 300.

    Returns:
      The length of the longest valid subsequence.
    """
    
    N = len(nums)
    # Problem guarantees N >= 2. 
    # The minimum possible length for a subsequence is 1 (any single element).
    # Since N >= 2, there is always a subsequence of length 2 (e.g., nums[0], nums[1]),
    # thus the minimal possible result for N >= 2 is 2.

    # Determine the maximum possible absolute difference between any two elements in nums.
    # The values in nums are between 1 and 300.
    # The maximum possible absolute difference is 300 - 1 = 299.
    # Let D_max = 299. The range of possible differences is [0, D_max].
    # We will use arrays indexed up to D_max, so their size needs to be D_max + 1 = 300.
    D_max = 299 
    
    # dp[i][diff] stores the length of the longest valid subsequence ending at index i
    # where the absolute difference between nums[i] and its predecessor in the subsequence is `diff`.
    # Initialize all lengths to 1. This represents the base case where the subsequence consists
    # of only the element nums[i]. Such a subsequence has length 1.
    # The size of the dp table is N x (D_max + 1).
    dp = [[1] * (D_max + 1) for _ in range(N)]
    
    # suffix_max_dp[i][d] stores the maximum value of dp[i][p] for all p >= d.
    # This auxiliary table helps optimize the lookup for the maximum length of a subsequence
    # ending at index i with a last difference of at least d.
    # Initialize all values to 1, consistent with the dp table initialization.
    # The size is also N x (D_max + 1).
    suffix_max_dp = [[1] * (D_max + 1) for _ in range(N)]
    
    # Keeps track of the overall maximum length found across all subsequences ending at any index.
    # Initialize to 1, as the minimum possible subsequence length is 1.
    # It will be updated throughout the process and will be at least 2 for N >= 2.
    max_len = 1
    
    # The base case for i=0 is already handled by the initialization.
    # dp[0][...] = 1 and suffix_max_dp[0][...] = 1 correctly reflect that any subsequence
    # starting and ending at index 0 has length 1.

    # Iterate through the array starting from the second element (index 1).
    for i in range(1, N):
        # For each element nums[i], consider all possible predecessors nums[k] where k < i.
        for k in range(i):
            # Calculate the absolute difference between nums[i] and nums[k].
            # This would be the last difference if nums[i] follows nums[k] in a subsequence.
            current_diff = abs(nums[i] - nums[k])
            
            # The calculated difference must be within the valid range [0, D_max].
            # Given the constraints 1 <= nums[i] <= 300, the difference is always between 0 and 299.
            # So, current_diff is guaranteed to be <= D_max.

            # To extend a subsequence ending at index k with nums[i], the validity condition is:
            # The previous difference `prev_diff` (difference before nums[k]) must be >= `current_diff`.
            # We need to find the maximum length of such a subsequence ending at k.
            # This maximum length is efficiently obtained from suffix_max_dp[k][current_diff].
            # Let max_prev_len be this maximum length.
            max_prev_len = suffix_max_dp[k][current_diff]
            
            # If we extend the subsequence ending at k (which has length max_prev_len) by appending nums[i],
            # the new subsequence has length max_prev_len + 1.
            # We update dp[i][current_diff] if this path yields a longer subsequence
            # ending at index i with the last difference being `current_diff`.
            # Note: since max_prev_len >= 1 (from initialization), the new length is at least 2.
            dp[i][current_diff] = max(dp[i][current_diff], max_prev_len + 1)

        # After considering all predecessors k for the current element nums[i],
        # the dp[i] row contains the maximum lengths for subsequences ending at i for each possible last difference.
        # Now, compute the suffix maximums for the dp[i] row. This information will be used
        # when processing subsequent elements nums[j] where j > i.
        # Compute suffix maximums by iterating from the largest difference D_max down to 0.
        suffix_max_dp[i][D_max] = dp[i][D_max]
        for d in range(D_max - 1, -1, -1):
            # The maximum length ending at i with difference >= d is the maximum of:
            # - The length ending at i with difference exactly d (dp[i][d])
            # - The maximum length ending at i with difference >= d+1 (suffix_max_dp[i][d+1])
            suffix_max_dp[i][d] = max(dp[i][d], suffix_max_dp[i][d + 1])
        
        # The maximum length of any valid subsequence ending at index i is the maximum value
        # across all possible last differences in the dp[i] row. This maximum value is
        # conveniently captured by suffix_max_dp[i][0].
        # Update the overall maximum length found so far.
        max_len = max(max_len, suffix_max_dp[i][0])
            
    # Return the overall maximum length found.
    return max_len