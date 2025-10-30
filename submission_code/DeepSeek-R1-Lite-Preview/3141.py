from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        N = len(nums)
        sum_nums = sum(nums)
        
        if sum_nums == 0:
            return -1  # Since nums[i] >= 1, sum_nums cannot be 0
        
        # Calculate the minimal number of full nums repetitions needed
        k = target // sum_nums
        if target % sum_nums != 0:
            k += 1
        # We consider up to k + 1 repetitions
        concatenated_nums = nums * (k + 1)
        
        min_length = float('inf')
        current_sum = 0
        start = 0
        
        for end in range(len(concatenated_nums)):
            current_sum += concatenated_nums[end]
            
            # Shrink the window from the start as long as the current_sum is greater than target
            while current_sum > target and start <= end:
                current_sum -= concatenated_nums[start]
                start += 1
            
            # Check if the current_sum is equal to target
            if current_sum == target:
                min_length = min(min_length, end - start + 1)
        
        return min_length if min_length != float('inf') else -1