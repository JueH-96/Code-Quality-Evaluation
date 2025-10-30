from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        max_score = 0
        
        def can_make_frequency(length):
            for i in range(n - length + 1):
                mid = i + length // 2
                target = nums[mid]
                cost = (target * (mid - i + 1) - (prefix_sum[mid + 1] - prefix_sum[i])) + \
                       ((prefix_sum[i + length] - prefix_sum[mid]) - target * (length - (mid - i + 1)))
                if cost <= k:
                    return True
            return False
        
        left, right = 1, n
        while left <= right:
            mid = (left + right) // 2
            if can_make_frequency(mid):
                max_score = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return max_score