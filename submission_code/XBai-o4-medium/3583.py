from typing import List
import bisect

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        if not nums:
            return []
        
        max_num = max(nums)
        freq = [0] * (max_num + 1)
        for num in nums:
            freq[num] += 1
        
        # Compute count_multiples for each d
        count_multiples = [0] * (max_num + 1)
        for d in range(1, max_num + 1):
            for multiple in range(d, max_num + 1, d):
                count_multiples[d] += freq[multiple]
        
        # Compute exact array
        exact = [0] * (max_num + 1)
        for d in range(max_num, 0, -1):
            c = count_multiples[d]
            exact_d = c * (c - 1) // 2
            multiple_d = 2 * d
            while multiple_d <= max_num:
                exact_d -= exact[multiple_d]
                multiple_d += d
            exact[d] = exact_d
        
        # Collect sorted_ds
        sorted_ds = []
        for d in range(1, max_num + 1):
            if exact[d] > 0:
                sorted_ds.append(d)
        sorted_ds.sort()
        
        # Compute prefix array
        prefix = []
        current = 0
        for d in sorted_ds:
            current += exact[d]
            prefix.append(current)
        
        # Process queries
        ans = []
        for q in queries:
            target = q + 1
            idx = bisect.bisect_left(prefix, target)
            ans.append(sorted_ds[idx])
        
        return ans