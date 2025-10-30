import bisect
from typing import List

class Solution:
    _pals = None
    
    def minimumCost(self, nums: List[int]) -> int:
        # Generate palindromes once as class variable
        if Solution._pals is None:
            pals = set()
            for length in range(1, 10):
                half_length = (length + 1) // 2
                start = 10 ** (half_length - 1)
                end = 10 ** half_length
                for prefix in range(start, end):
                    s = str(prefix)
                    if length % 2 == 0:
                        pal_str = s + s[::-1]
                    else:
                        pal_str = s + s[:-1][::-1]
                    pal = int(pal_str)
                    if pal < 10**9:
                        pals.add(pal)
            Solution._pals = list(pals)
        pal_list = Solution._pals
        
        # Process nums
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums_sorted[i]
        min_cost = float('inf')
        for y in pal_list:
            k = bisect.bisect_right(nums_sorted, y)
            cost = y * k - prefix_sum[k] + (prefix_sum[n] - prefix_sum[k] - y * (n - k))
            if cost < min_cost:
                min_cost = cost
        return min_cost