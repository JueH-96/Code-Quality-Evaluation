from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Check all possible subarrays
        for start in range(n):
            for end in range(start + 1, n + 1):
                # Create a new array by removing the subarray nums[start:end]
                new_nums = nums[:start] + nums[end:]
                
                # Check if the new array is strictly increasing
                is_strictly_increasing = all(new_nums[i] < new_nums[i + 1] for i in range(len(new_nums) - 1))
                
                if is_strictly_increasing:
                    count += 1
        
        return count