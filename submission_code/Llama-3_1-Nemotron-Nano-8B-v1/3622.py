from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        max_freq = 0
        left = 0
        
        for right in range(n):
            # Ensure the current window has range <= 2*k
            while nums[right] - nums[left] > 2 * k:
                left += 1
            
            # Binary search to find the best left' for the current right
            low = left
            high = right
            best_left = left
            while low <= high:
                mid = (low + high) // 2
                window_length = right - mid + 1
                m = mid + (window_length - 1) // 2  # median index
                # Calculate sum of absolute differences from the median
                sum_left = (m - mid) * nums[m] - (prefix[m] - prefix[mid])
                sum_right = (prefix[right + 1] - prefix[m + 1]) - (right - m) * nums[m]
                total = sum_left + sum_right
                
                if total <= numOperations * k:
                    best_left = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            current_freq = right - best_left + 1
            if current_freq > max_freq:
                max_freq = current_freq
        
        return max_freq