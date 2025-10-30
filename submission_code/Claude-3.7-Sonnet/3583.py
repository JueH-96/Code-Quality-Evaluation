from math import gcd
from typing import List
from collections import Counter
import bisect

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Calculate GCD of all possible pairs and count their frequencies
        n = len(nums)
        gcd_counter = Counter()
        for i in range(n):
            for j in range(i + 1, n):
                gcd_counter[gcd(nums[i], nums[j])] += 1
        
        # Sort the GCD values
        sorted_gcd = sorted(gcd_counter.items())
        
        # Calculate cumulative frequencies
        cumulative_freq = []
        current_sum = 0
        for g, freq in sorted_gcd:
            current_sum += freq
            cumulative_freq.append(current_sum)
        
        # Answer the queries using binary search
        result = []
        for q in queries:
            idx = bisect.bisect_left(cumulative_freq, q + 1)
            if idx < len(sorted_gcd):
                result.append(sorted_gcd[idx][0])
            else:
                # This should not happen given the constraints, but just to be safe
                result.append(sorted_gcd[-1][0])
        
        return result