from bisect import bisect_right
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        if not nums:
            return []
        max_num = max(nums)
        freq = [0] * (max_num + 1)
        for num in nums:
            freq[num] += 1
        
        exact = [0] * (max_num + 2)  # To handle up to max_num * 2
        
        for d in range(max_num, 0, -1):
            # Calculate the number of elements divisible by d
            cnt = 0
            multiple = d
            while multiple <= max_num:
                cnt += freq[multiple]
                multiple += d
            # Calculate pairs
            pairs = cnt * (cnt - 1) // 2
            # Subtract exact counts of multiples of d greater than d
            subtract = 0
            multiple = 2 * d
            while multiple <= max_num:
                subtract += exact[multiple]
                multiple += d
            exact[d] = pairs - subtract
        
        # Collect and sort the GCDs with non-zero counts
        sorted_gcds = []
        for d in range(1, max_num + 1):
            if exact[d] > 0:
                sorted_gcds.append(d)
        sorted_gcds.sort()
        
        # Build the prefix sum array
        prefix = [0]
        current = 0
        for d in sorted_gcds:
            current += exact[d]
            prefix.append(current)
        
        # Process each query using binary search
        result = []
        for k in queries:
            idx = bisect_right(prefix, k)
            result.append(sorted_gcds[idx - 1])
        
        return result