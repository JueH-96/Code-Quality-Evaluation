from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # Find the maximum value in nums
        max_num = max(nums)
        # Define the range for the difference array
        max_value = max_num + k
        # Initialize the difference array with zeros
        diff = [0] * (max_value + 2)
        
        # Iterate through each number in nums
        for num in nums:
            low = max(0, num - k)
            high = num + k
            # Increment the start of the range
            diff[low] += 1
            # Decrement just after the end of the range
            if high + 1 < len(diff):
                diff[high + 1] -= 1
        
        # Compute the prefix sums to get the coverage counts
        max_beauty = 0
        current = 0
        for value in diff:
            current += value
            if current > max_beauty:
                max_beauty = current
        
        return max_beauty