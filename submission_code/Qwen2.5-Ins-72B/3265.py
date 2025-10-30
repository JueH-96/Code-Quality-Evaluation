from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = float('-inf')
        prefix_sum = [0]
        num_to_index = {}
        
        for i, num in enumerate(nums):
            prefix_sum.append(prefix_sum[-1] + num)
            
            if num - k in num_to_index:
                start_index = num_to_index[num - k]
                max_sum = max(max_sum, prefix_sum[i + 1] - prefix_sum[start_index])
            
            if num + k in num_to_index:
                start_index = num_to_index[num + k]
                max_sum = max(max_sum, prefix_sum[i + 1] - prefix_sum[start_index])
            
            if num not in num_to_index:
                num_to_index[num] = i
            else:
                num_to_index[num] = min(num_to_index[num], i)
        
        return max_sum if max_sum != float('-inf') else 0