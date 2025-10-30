from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # The element we have to track – the maximum in the whole array
        target = max(nums)
        
        # Helper: number of sub-arrays that contain the target **at most** t times
        def at_most(t: int) -> int:
            left = 0
            cnt_target = 0
            res = 0
            for right, val in enumerate(nums):
                if val == target:
                    cnt_target += 1
                while cnt_target > t:
                    if nums[left] == target:
                        cnt_target -= 1
                    left += 1
                # all subarrays ending at `right` and starting between `left`..`right`
                res += right - left + 1
            return res
        
        total_subarrays = n * (n + 1) // 2
        # subarrays with at most k-1 occurrences of the maximum
        without_enough_max = at_most(k - 1)
        return total_subarrays - without_enough_max