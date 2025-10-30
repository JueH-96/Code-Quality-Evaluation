from typing import List

class Solution:
  def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
    n = len(nums)
    nums.sort() # Sorts in-place, O(N log N) time
    
    operations = 0
    
    # The median element is at index n // 2 after sorting.
    # Let median_idx = n // 2.
    
    # Part 1: Ensure elements up to and including the median are <= k.
    # Iterate through the first half of the array, including the median element (index median_idx).
    # These elements (nums[i] for i <= median_idx) must be made <= k.
    # If nums[i] > k, it must be reduced to k. The cost for this is nums[i] - k.
    median_idx = n // 2
    for i in range(median_idx + 1): # Loop from index 0 up to median_idx (inclusive)
      if nums[i] > k:
        operations += nums[i] - k
        
    # Part 2: Ensure elements from the median onwards are >= k.
    # Iterate through the second half of the array, including the median element (index median_idx).
    # These elements (nums[i] for i >= median_idx) must be made >= k.
    # If nums[i] < k, it must be increased to k. The cost for this is k - nums[i].
    for i in range(median_idx, n): # Loop from index median_idx up to n-1 (inclusive)
      if nums[i] < k:
        operations += k - nums[i]
            
    # Note on nums[median_idx]:
    # - If nums[median_idx] > k: The first loop adds (nums[median_idx] - k). The condition in the second loop (nums[median_idx] < k) is false.
    # - If nums[median_idx] < k: The condition in the first loop (nums[median_idx] > k) is false. The second loop adds (k - nums[median_idx]).
    # - If nums[median_idx] == k: Neither condition is met.
    # This correctly accumulates abs(nums[median_idx] - k) for the median element itself.
    # Elements nums[i] where i < median_idx are only affected by the first loop.
    # Elements nums[i] where i > median_idx are only affected by the second loop.
            
    return operations