from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        low, high = 1, n
        res = 0
        
        while low <= high:
            mid = (low + high) // 2
            found = False
            
            # Check all possible windows of size 'mid'
            for i in range(n - mid + 1):
                end = i + mid - 1
                m = i + (mid - 1) // 2  # median index in current window
                median_val = nums[m]
                
                # Calculate sum of absolute differences to median
                sum_left = median_val * (m - i) - (prefix[m] - prefix[i])
                sum_right = (prefix[end + 1] - prefix[m + 1]) - median_val * (end - m)
                total_cost = sum_left + sum_right
                
                if total_cost <= k:
                    found = True
                    break  # No need to check other windows once found
            
            if found:
                res = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return res