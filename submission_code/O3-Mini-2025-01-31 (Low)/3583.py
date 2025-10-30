from typing import List
import math
import bisect

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Get maximum possible value from nums for bounds
        max_val = max(nums)
        # Build frequency array for numbers [0..max_val]
        freq = [0] * (max_val + 1)
        for num in nums:
            freq[num] += 1
        
        # For each candidate d from 1..max_val,
        # we want to count how many numbers in nums are multiples of d.
        multiplesCount = [0] * (max_val + 1)
        for d in range(1, max_val + 1):
            # Sum over multiples of d: d, 2d, 3d, ...
            for multiple in range(d, max_val + 1, d):
                multiplesCount[d] += freq[multiple]
        
        # Let gcd_freq[d] be the count of pairs with gcd exactly d.
        # using inclusion-exclusion starting from larger d multiples.
        gcd_freq = [0] * (max_val + 1)
        # We iterate d descending, because we want to subtract from lower gcd values.
        for d in range(max_val, 0, -1):
            # count pairs with both numbers divisible by d.
            total = multiplesCount[d]
            # Number of pairs that have gcd a multiple of d.
            total_pairs = total * (total - 1) // 2
            # Subtract those pairs that have gcd higher than d (i.e. multiples of d > d)
            j = 2 * d
            j_sum = 0
            while j <= max_val:
                total_pairs -= gcd_freq[j]
                j += d
            gcd_freq[d] = total_pairs
        
        # Now, every pair's gcd is computed: if gcd_freq[d] > 0, we have that many occurrences of d.
        # We need to answer queries on the sorted array of gcd pairs, where the array is sorted in ascending order.
        # Notice: In the sorted order, the smallest values appear first.
        # Thus, we want to build an array of (g, frequency) in ascending order of g.
        gcd_values = []
        # accumulate prefix sums to binary search on queries.
        prefix = []
        current_sum = 0
        for d in range(1, max_val + 1):
            if gcd_freq[d] > 0:
                gcd_values.append(d)
                current_sum += gcd_freq[d]
                prefix.append(current_sum)
        
        # Now answer the queries
        # Each query q is an index (0-indexed) into the sorted gcdPairs array.
        # We want to find the smallest index in prefix with value > q.
        res = []
        for q in queries:
            # binary search: find leftmost prefix index where prefix[i] > q.
            # Because q is index (0-indexed) so if prefix[i] > q, then gcdValues[i] is the answer.
            idx = bisect.bisect_right(prefix, q)
            # Actually, bisect_right(prefix, q) works if prefix[i-1] <= q < prefix[i].
            # idx is the first position where prefix value > q.
            res.append(gcd_values[idx])
        return res