from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        prefix_sum = {0: -1}
        curr_sum = 0
        min_length = float('inf')
        
        for i in range(n):
            curr_sum += nums[i]
            if curr_sum - target in prefix_sum:
                min_length = min(min_length, i - prefix_sum[curr_sum - target])
            if curr_sum not in prefix_sum:
                prefix_sum[curr_sum] = i
        
        if min_length == float('inf'):
            return -1
        
        # Check for subarrays that span across the boundary of the original array
        for i in range(n):
            curr_sum = 0
            for j in range(i, i + n):
                curr_sum += nums[j % n]
                if curr_sum == target:
                    min_length = min(min_length, j - i + 1)
        
        return min_length