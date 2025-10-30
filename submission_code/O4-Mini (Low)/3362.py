from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        # Total number of subarrays
        total = n * (n + 1) // 2
        # For even, we take the lower median index
        target = (total + 1) // 2 if total % 2 else total // 2
        
        # Count total distinct in the entire array as upper bound
        distinct_all = len(set(nums))
        
        # Helper: count of subarrays with at most K distinct elements
        def at_most_k(K: int) -> int:
            count = {}
            distinct = 0
            left = 0
            res = 0
            for right, v in enumerate(nums):
                count[v] = count.get(v, 0) + 1
                if count[v] == 1:
                    distinct += 1
                # shrink window until at most K distinct
                while distinct > K:
                    x = nums[left]
                    count[x] -= 1
                    if count[x] == 0:
                        distinct -= 1
                    left += 1
                # all subarrays ending at right with >= left start are valid
                res += (right - left + 1)
            return res
        
        # We want the smallest k such that
        # number of subarrays with exactly k distinct
        # cumulative up to k >= target
        # But that cumulative is f(k) = at_most_k(k)
        lo, hi = 1, distinct_all
        ans = distinct_all
        while lo <= hi:
            mid = (lo + hi) // 2
            if at_most_k(mid) >= target:
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        return ans