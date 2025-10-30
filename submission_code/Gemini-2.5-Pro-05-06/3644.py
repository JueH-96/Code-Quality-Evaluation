from typing import List

class Solution:
  def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
    n = len(nums)
    min_positive_sum = float('inf') # Initialize with a value larger than any possible sum

    # Iterate over all possible start indices i of a subarray
    for i in range(n):
      current_sum = 0
      # Iterate over all possible end indices j of a subarray, starting from i
      # This ensures that j >= i, forming valid subarrays nums[i...j]
      for j in range(i, n):
        # Add the current element nums[j] to the sum of the subarray nums[i...j]
        current_sum += nums[j]
        
        # Calculate the length of the current subarray nums[i...j]
        length = j - i + 1
        
        # Optimization: If current subarray length already exceeds r,
        # then any further subarrays starting at i (by extending j) will also be too long.
        # So, we can break from the inner loop for this i.
        if length > r:
          break 
        
        # Check if current subarray length is at least l.
        # (We already know length <= r due to the break condition above, unless r was already passed initially)
        # This effectively checks if l <= length <= r.
        if length >= l:
          # If the length is valid, check if the sum of the subarray is positive
          if current_sum > 0:
            # If the sum is positive, update min_positive_sum if current_sum is smaller
            min_positive_sum = min(min_positive_sum, current_sum)
            
    # After checking all valid subarrays, if min_positive_sum is still float('inf'),
    # it means no subarray satisfying all conditions was found.
    if min_positive_sum == float('inf'):
      return -1
    else:
      # Otherwise, return the minimum positive sum found.
      # Since all elements in nums are integers, current_sum will always be an integer.
      # So, min_positive_sum (if not inf) will be an integer.
      return int(min_positive_sum)