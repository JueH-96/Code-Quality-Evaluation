from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        global_max = max(nums)
        n = len(nums)
        total = n * (n + 1) // 2
        
        def atMost(m):
            left = 0
            count = 0
            res = 0
            for right in range(n):
                if nums[right] == global_max:
                    count += 1
                # Shrink the window until count <= m
                while count > m:
                    if nums[left] == global_max:
                        count -= 1
                    left += 1
                # Add the number of valid subarrays ending at right
                res += right - left + 1
            return res
        
        m = k - 1
        x = atMost(m)
        return total - x