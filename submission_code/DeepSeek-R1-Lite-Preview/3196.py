from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        def is_possible(m):
            for left in range(n - m + 1):
                right = left + m - 1
                median_pos = left + m // 2
                median = nums[median_pos]
                
                sum_left = median * (median_pos - left) - (prefix_sum[median_pos] - prefix_sum[left])
                sum_right = (prefix_sum[right + 1] - prefix_sum[median_pos + 1]) - median * (right - median_pos)
                
                total_operations = sum_left + sum_right
                if total_operations <= k:
                    return True
            return False
        
        low, high = 1, n
        result = 1
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                result = mid
                low = mid + 1
            else:
                high = mid - 1
        return result