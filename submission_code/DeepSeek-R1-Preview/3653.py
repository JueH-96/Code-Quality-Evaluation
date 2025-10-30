from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        mod_dict = {0: 0}
        max_sum = float('-inf')
        
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
            r = i % k
            
            if r in mod_dict:
                current_sum = prefix[i] - mod_dict[r]
                if current_sum > max_sum:
                    max_sum = current_sum
            
            # Update mod_dict with the minimum prefix sum for the current remainder
            if r in mod_dict:
                if prefix[i] < mod_dict[r]:
                    mod_dict[r] = prefix[i]
            else:
                mod_dict[r] = prefix[i]
        
        return max_sum