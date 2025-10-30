from typing import List

class Solution:
  def missingInteger(self, nums: List[int]) -> int:
    # Step 1: Find the sum of the longest sequential prefix
    
    # Constraints: 1 <= nums.length <= 50, so nums is never empty.
    # The prefix nums[0...0] (i.e., just nums[0]) is always sequential.
    current_prefix_sum = nums[0]

    # Iterate from the second element to extend the sequential prefix.
    # If len(nums) is 1, this loop does not run, which is correct;
    # current_prefix_sum will remain nums[0].
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            current_prefix_sum += nums[i]
        else:
            # The sequential prefix ends at nums[i-1].
            # Stop extending the sum.
            break
    
    # current_prefix_sum now holds the sum of the longest sequential prefix.

    # Step 2: Find the smallest integer x >= current_prefix_sum 
    #         that is not in nums.
    
    # Convert nums to a set for O(1) average time complexity for membership testing.
    num_set = set(nums)

    # Start checking from x = current_prefix_sum.
    # We are looking for the smallest integer `x` such that:
    #   1. x >= current_prefix_sum
    #   2. x is not in num_set (i.e., missing from nums)
    
    x = current_prefix_sum
    while True: # This loop will eventually terminate.
        if x not in num_set:
            # Found the smallest x satisfying the conditions.
            return x
        
        # If x is in num_set, try the next integer.
        x += 1
        
        # This loop iterates at most len(nums) + 1 times in a worst-case scenario
        # (e.g., if current_prefix_sum and the next len(nums)-1 consecutive integers
        # are all present in num_set). Since len(nums) <= 50, this is efficient.