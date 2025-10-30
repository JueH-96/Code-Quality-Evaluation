from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        low, high = 1, n
        best = 0
        
        while low <= high:
            mid_L = (low + high) // 2
            found = False
            
            max_left = n - mid_L
            for left in range(max_left + 1):
                right = left + mid_L - 1
                mid = left + (mid_L - 1) // 2
                
                sum_left = prefix[mid + 1] - prefix[left]
                sum_right = prefix[right + 1] - prefix[mid + 1]
                
                cost = (nums[mid] * (mid - left + 1) - sum_left) + (sum_right - nums[mid] * (right - mid))
                
                if cost <= k:
                    found = True
                    break  # Found a valid window, no need to check further
            
            if found:
                best = mid_L
                low = mid_L + 1
            else:
                high = mid_L - 1
        
        return best