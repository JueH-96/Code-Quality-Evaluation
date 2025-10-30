import collections # collections is not actually used, but safe to keep import if unsure
from typing import List # Import List type hint

class Solution:
  """
  Finds the smallest integer missing from nums such that it's greater than or equal
  to the sum of the longest sequential prefix of nums.
  """
  def missingInteger(self, nums: List[int]) -> int:
    """
    Calculates the sum of the longest sequential prefix and then finds the 
    smallest integer >= sum that is not present in nums.

    Args:
      nums: A 0-indexed array of integers. The constraints guarantee that
            nums is not empty (1 <= nums.length <= 50) and contains integers
            between 1 and 50 (1 <= nums[i] <= 50).

    Returns:
      The smallest missing integer greater than or equal to the sum of the
      longest sequential prefix.
    """

    # Constraints state nums.length >= 1, so nums[0] is always accessible.
    # Initialize prefix_sum with the first element, as a single element prefix is always sequential.
    prefix_sum = nums[0]
    
    # Step 1 & 2 combined: Find the longest sequential prefix and calculate its sum.
    # Iterate through the array starting from the second element (index 1).
    for i in range(1, len(nums)):
        # Check if the current element forms a sequence with the previous one.
        # A sequence means nums[i] must be exactly one greater than nums[i-1].
        if nums[i] == nums[i - 1] + 1:
            # If it continues the sequence, add the current element to the prefix sum.
            prefix_sum += nums[i]
        else:
            # If the sequence breaks (nums[i] != nums[i-1] + 1), 
            # the longest sequential prefix ends at index i-1.
            # Stop calculating the sum.
            break
    
    # Step 3: Find the smallest integer x >= prefix_sum that is missing from nums.
    # Convert the input list `nums` into a set for efficient membership checking.
    # Checking if an element is in a set takes O(1) average time complexity.
    nums_set = set(nums)
    
    # Initialize the number to check with the calculated prefix sum.
    current_check = prefix_sum
    
    # Keep incrementing `current_check` as long as it is present in the `nums_set`.
    # This loop finds the first integer value, starting from `prefix_sum`, 
    # that is *not* present in the original `nums` array.
    while current_check in nums_set:
        current_check += 1
            
    # The loop terminates when `current_check` holds the smallest integer 
    # value that is >= `prefix_sum` and is not found in `nums`. This is the required result.
    return current_check