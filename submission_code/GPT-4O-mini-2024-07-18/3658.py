from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        # Extract the known numbers and their positions
        known_nums = [num for num in nums if num != -1]
        if not known_nums:
            return 0  # All numbers are -1, we can choose any pair (x, y) and make them equal
        
        # Find the minimum and maximum of the known numbers
        min_known = min(known_nums)
        max_known = max(known_nums)
        
        # If there are no missing values, return the maximum difference
        if len(known_nums) == len(nums):
            return max_known - min_known
        
        # Calculate the potential minimum difference
        # We can replace -1 with two numbers x and y
        # The best candidates for x and y are around the known numbers
        # We can try to minimize the maximum difference by choosing x and y
        # as close to the min and max of known numbers as possible
        
        # The candidates for x and y can be:
        # 1. Just below min_known
        # 2. Just above max_known
        # 3. Between min_known and max_known
        
        # We will consider the following pairs:
        candidates = [
            (min_known - 1, min_known - 1),  # Both x and y just below min_known
            (max_known + 1, max_known + 1),  # Both x and y just above max_known
            (min_known - 1, max_known + 1),  # x just below min_known, y just above max_known
            (min_known, max_known),            # x = min_known, y = max_known
            (min_known + 1, max_known),        # x just above min_known, y = max_known
            (min_known, max_known - 1),        # x = min_known, y just below max_known
        ]
        
        min_diff = float('inf')
        
        for x, y in candidates:
            # Replace -1 with x or y and calculate the maximum difference
            modified_nums = []
            for num in nums:
                if num == -1:
                    modified_nums.append(x)  # Replace with x
                else:
                    modified_nums.append(num)
            
            # Calculate max difference with x
            max_diff_x = max(abs(modified_nums[i] - modified_nums[i + 1]) for i in range(len(modified_nums) - 1))
            min_diff = min(min_diff, max_diff_x)
            
            # Replace -1 with y and calculate the maximum difference
            modified_nums = []
            for num in nums:
                if num == -1:
                    modified_nums.append(y)  # Replace with y
                else:
                    modified_nums.append(num)
            
            # Calculate max difference with y
            max_diff_y = max(abs(modified_nums[i] - modified_nums[i + 1]) for i in range(len(modified_nums) - 1))
            min_diff = min(min_diff, max_diff_y)
        
        return min_diff