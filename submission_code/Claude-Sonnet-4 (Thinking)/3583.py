from collections import Counter
import bisect

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        max_val = max(nums)
        
        # Count frequency of each number
        freq = Counter(nums)
        
        # Count how many numbers are multiples of each value
        multiples_count = [0] * (max_val + 1)
        for i in range(1, max_val + 1):
            for j in range(i, max_val + 1, i):
                multiples_count[i] += freq[j]
        
        # Count pairs with GCD exactly equal to i using inclusion-exclusion
        gcd_count = [0] * (max_val + 1)
        for i in range(max_val, 0, -1):
            count = multiples_count[i]
            gcd_count[i] = count * (count - 1) // 2
            
            # Subtract pairs with GCD that is a multiple of i (but not i itself)
            for j in range(2 * i, max_val + 1, i):
                gcd_count[i] -= gcd_count[j]
        
        # Build cumulative counts
        cumulative = [0]
        gcd_values = []
        for gcd_val in range(1, max_val + 1):
            if gcd_count[gcd_val] > 0:
                gcd_values.append(gcd_val)
                cumulative.append(cumulative[-1] + gcd_count[gcd_val])
        
        # Answer queries using binary search
        result = []
        for q in queries:
            # Find the GCD value for the q-th pair (0-indexed)
            idx = bisect.bisect_right(cumulative, q)
            result.append(gcd_values[idx - 1])
        
        return result