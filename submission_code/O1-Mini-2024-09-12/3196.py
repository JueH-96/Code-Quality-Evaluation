from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        left = 0
        result = 1
        
        for right in range(n):
            # Calculate median index
            m = (left + right) // 2
            y = nums[m]
            
            # Sum of differences to the median
            sum_left = y * (m - left + 1) - (prefix[m + 1] - prefix[left])
            sum_right = (prefix[right + 1] - prefix[m + 1]) - y * (right - m)
            total = sum_left + sum_right
            
            # Shrink window from the left if total exceeds k
            while total > k and left <= right:
                left += 1
                if left > m:
                    m = (left + right) // 2
                    y = nums[m]
                # Recalculate sums after moving left
                sum_left = y * (m - left + 1) - (prefix[m + 1] - prefix[left])
                sum_right = (prefix[right + 1] - prefix[m + 1]) - y * (right - m)
                total = sum_left + sum_right
            
            # Update the result with the maximum window size
            result = max(result, right - left + 1)
        
        return result