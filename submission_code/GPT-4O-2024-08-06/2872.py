from typing import List

class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        # We will process the array from right to left
        # This allows us to accumulate sums in a greedy manner
        max_value = 0
        current_sum = 0
        
        # Traverse the array from the end to the start
        for num in reversed(nums):
            # If current number is less than or equal to the current sum,
            # we can merge it into the current sum
            if num <= current_sum:
                current_sum += num
            else:
                # Otherwise, start a new sum with the current number
                current_sum = num
            
            # Update the maximum value found so far
            max_value = max(max_value, current_sum)
        
        return max_value