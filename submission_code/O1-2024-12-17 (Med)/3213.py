from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0

        # Find the global maximum element in nums
        M = max(nums)
        
        # Count how many times M appears in the array
        count_M = sum(1 for x in nums if x == M)
        
        # If M appears fewer than k times in total, answer is 0
        if count_M < k:
            return 0
        
        # Total number of subarrays in an array of length n
        total_subarrays = n * (n + 1) // 2
        
        # Helper function: number of subarrays that contain the element M at most t times
        def subAtMost(t: int) -> int:
            if t < 0:
                return 0
            
            left = 0
            freq = 0  # how many times M appears in the current window
            result = 0
            
            for right in range(n):
                if nums[right] == M:
                    freq += 1
                    
                while freq > t:
                    if nums[left] == M:
                        freq -= 1
                    left += 1
                    
                result += (right - left + 1)
            
            return result
        
        # Number of subarrays with at least k occurrences of M
        # = Total subarrays - number of subarrays with at most (k-1) occurrences of M
        return total_subarrays - subAtMost(k - 1)