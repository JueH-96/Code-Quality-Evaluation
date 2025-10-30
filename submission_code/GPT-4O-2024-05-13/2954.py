from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if k > n:
            return 0
        
        max_sum = 0
        current_sum = 0
        window = {}
        
        for i in range(k):
            current_sum += nums[i]
            if nums[i] in window:
                window[nums[i]] += 1
            else:
                window[nums[i]] = 1
        
        if len(window) >= m:
            max_sum = current_sum
        
        for i in range(k, n):
            current_sum += nums[i] - nums[i - k]
            
            if nums[i] in window:
                window[nums[i]] += 1
            else:
                window[nums[i]] = 1
            
            window[nums[i - k]] -= 1
            if window[nums[i - k]] == 0:
                del window[nums[i - k]]
            
            if len(window) >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum