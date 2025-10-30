from typing import List
from collections import defaultdict

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        # median index (1-indexed) in sorted order
        k = (total_subarrays + 1) // 2
        
        # Helper function: counts the total number of subarrays with at most K distinct elements.
        def countAtMost(K: int) -> int:
            count = 0
            left = 0
            freq = defaultdict(int)
            distinct = 0
            for right in range(n):
                # include nums[right]
                if freq[nums[right]] == 0:
                    distinct += 1
                freq[nums[right]] += 1
                # shrink window until distinct <= K
                while distinct > K:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        distinct -= 1
                    left += 1
                # For each right, subarrays ending at right with at most K distinct are:
                count += (right - left + 1)
            return count
        
        # The maximum possible distinct count in any subarray is the distinct elements in the whole array.
        maxDistinct = len(set(nums))
        
        # Binary search for the smallest candidate d (in [1, maxDistinct]) 
        # such that countAtMost(d) is at least k.
        lo, hi = 1, maxDistinct
        ans = hi
        while lo <= hi:
            mid = (lo + hi) // 2
            if countAtMost(mid) >= k:
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans