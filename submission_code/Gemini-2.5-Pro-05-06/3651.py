from typing import List

class Solution:
  def constructTransformedArray(self, nums: List[int]) -> List[int]:
    n = len(nums)
    
    # Constraints: 1 <= nums.length <= 100.
    # This means n is always at least 1.
    # No need to handle an empty list case explicitly based on constraints.
    
    result = [0] * n  # Initialize the result array.
                      # Each element result[i] will be computed and assigned in the loop.
    
    for i in range(n):
      current_element_value = nums[i]  # This is the value from nums[i].
                                       # It dictates the number of steps and direction.
      
      if current_element_value == 0:
        # Rule: If nums[i] == 0, set result[i] to nums[i].
        # Since nums[i] is 0 in this case, result[i] is set to 0.
        result[i] = 0
      else:
        # This block handles cases where nums[i] is non-zero (either positive or negative).
        
        # If nums[i] > 0: move nums[i] steps to the right.
        # If nums[i] < 0: move abs(nums[i]) steps to the left.
        # Both scenarios are covered by a net displacement of current_element_value from index i.
        # A positive displacement means moving right; a negative displacement means moving left.
        
        # Calculate the landing_index in the circular array.
        # The formula is (start_index + displacement) % array_length.
        landing_index = (i + current_element_value) % n
        
        # Python's % (modulo) operator behavior is suitable for circular arrays:
        # - The result of 'x % n' (where n is positive) is always in the range [0, n-1]
        #   (or 0 if x is a multiple of n, which is also correct for 0-indexed arrays).
        # - This correctly handles cases where (i + current_element_value) is negative.
        #   For example, if i=0, current_element_value=-1, n=4:
        #   landing_index = (0 + (-1)) % 4 = -1 % 4 = 3.
        
        # Rule: "Set result[i] to the value of the index where you land."
        # The problem's examples clarify this means result[i] = nums[landing_index].
        # For example, in Example 1 (nums = [3,-2,1,1]):
        #   For nums[0]=3: Start at index 0, move 3 steps right.
        #   Landing index is (0+3)%4 = 3.
        #   The explanation states: "reach nums[3]. So result[0] should be 1."
        #   Since nums[3] is 1, this confirms result[0] = nums[landing_index].
        result[i] = nums[landing_index]
            
    return result