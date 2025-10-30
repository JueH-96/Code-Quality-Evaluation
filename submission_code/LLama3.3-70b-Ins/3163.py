from typing import List

class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        
        # Generate all possible subarrays
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                
                # Calculate the distinct count of the subarray
                distinct_count = len(set(subarray))
                
                # Add the square of the distinct count to the total sum
                total_sum += distinct_count ** 2
        
        return total_sum