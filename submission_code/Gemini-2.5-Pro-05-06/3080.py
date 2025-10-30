from typing import List

class Solution:
  def maxSubarrays(self, nums: List[int]) -> int:
    n = len(nums)
    
    # Calculate the bitwise AND of all elements in the array.
    # Let this be S_all.
    s_all = nums[0]
    for i in range(1, n):
      s_all &= nums[i]
      
    # Case 1: S_all > 0
    # The minimum possible sum of scores is S_all.
    # To achieve this sum, if S_all > 0, we must have exactly one subarray (the whole array).
    # Any split into k > 1 subarrays would result in a sum of scores >= k * S_all > S_all.
    if s_all > 0:
      return 1
      
    # Case 2: S_all == 0
    # The minimum possible sum of scores is 0.
    # This requires every subarray in the partition to have a score of 0.
    # We want to maximize the number of such subarrays.
    # We use a greedy approach: make each subarray as short as possible.
    
    count = 0
    # Initialize current_segment_and. Using -1 in Python for signed integers
    # means all bits are effectively 1s in two's complement.
    # (-1 & X) results in X for non-negative X. So -1 acts as an identity for AND.
    current_segment_and = -1 
    
    for x in nums:
      current_segment_and &= x
      if current_segment_and == 0:
        # The current segment (from its start to x) has an AND sum of 0.
        # This forms one subarray in our partition.
        count += 1
        # Reset current_segment_and to start a new segment.
        current_segment_and = -1 
        
    # If S_all == 0, the entire array's AND sum is 0.
    # The greedy process ensures that `current_segment_and` (if never reset)
    # would become 0 at the end of the array processing, incrementing `count` at least once.
    # Thus, `count` will be at least 1 if S_all == 0.
    return count