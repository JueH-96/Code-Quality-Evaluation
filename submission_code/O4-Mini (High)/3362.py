from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        # Total number of subarrays
        total_subarrays = n * (n + 1) // 2
        # 0-based median index in sorted uniqueness array
        m = (total_subarrays - 1) // 2
        
        # Precompute the maximum value to size our count array
        maxv = max(nums)
        # Also compute how many distinct values appear in the entire array
        unique_count = len(set(nums))
        
        # Helper: number of subarrays with at most d distinct elements
        def at_most(d: int) -> int:
            count = [0] * (maxv + 1)
            left = 0
            distinct = 0
            res = 0
            for right, v in enumerate(nums):
                if count[v] == 0:
                    distinct += 1
                count[v] += 1
                # shrink window until we have <= d distinct
                while distinct > d:
                    x = nums[left]
                    count[x] -= 1
                    if count[x] == 0:
                        distinct -= 1
                    left += 1
                # all subarrays ending at right with start in [left..right] are valid
                res += (right - left + 1)
            return res
        
        # Binary search for the smallest d in [1..unique_count]
        # such that #subarrays with <= d distinct > m
        lo, hi = 1, unique_count
        while lo < hi:
            mid = (lo + hi) // 2
            if at_most(mid) > m:
                hi = mid
            else:
                lo = mid + 1
        return lo