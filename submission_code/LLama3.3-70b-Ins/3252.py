from typing import List

class Solution:
    def incremovableSubarrayCount(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Generate all possible subarrays
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                
                # Remove the subarray from the original array
                new_nums = nums[:i] + nums[j+1:]
                
                # Check if the new array is strictly increasing
                if len(new_nums) == 0 or all(new_nums[k] < new_nums[k+1] for k in range(len(new_nums)-1)):
                    count += 1
                    
        return count