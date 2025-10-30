from typing import List
import math
import bisect

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # max value in nums
        max_val = max(nums)
        # frequency array for values from 1 to max_val
        freq = [0] * (max_val + 1)
        for num in nums:
            freq[num] += 1

        # Count how many numbers are divisible by d for each d = 1 .. max_val.
        divCount = [0] * (max_val + 1)
        for d in range(1, max_val + 1):
            # Sum frequency for multiples of d.
            for multiple in range(d, max_val + 1, d):
                divCount[d] += freq[multiple]

        # pairCount[d] counts pairs (i, j) with both numbers divisible by d.
        pairCount = [0] * (max_val + 1)
        for d in range(1, max_val + 1):
            if divCount[d] >= 2:
                pairCount[d] = divCount[d] * (divCount[d] - 1) // 2

        # f[d] will store count of pairs with exact gcd = d.
        f = [0] * (max_val + 1)
        # Process d from high to low
        for d in range(max_val, 0, -1):
            # Start with all pairs divisible by d.
            f[d] = pairCount[d]
            # Subtract pairs with gcd being a multiple of d > d.
            mult = 2 * d
            while mult <= max_val:
                f[d] -= f[mult]
                mult += d

        # Now, gcdPairs sorted in ascending order consists of:
        # f[1] copies of 1, f[2] copies of 2, ..., f[max_val] copies of max_val
        # Build cumulative frequency list for binary search.
        # We'll store a list of tuples (g, cumulative_count) for each g where f[g] > 0.
        cumGcd = []
        cum_sum = 0
        for g in range(1, max_val + 1):
            if f[g] > 0:
                cum_sum += f[g]
                cumGcd.append((g, cum_sum))

        # For answering queries: using binary search on the cumulative counts.
        res = []
        for q in queries:
            # q is 0-indexed. We want the element at position index q in the sorted array.
            # Find the first tuple where cumulative_count > q.
            # Use bisect_left on the second element.
            lo = 0
            hi = len(cumGcd)
            while lo < hi:
                mid = (lo + hi) // 2
                if cumGcd[mid][1] > q:
                    hi = mid
                else:
                    lo = mid + 1
            # cumGcd[lo] is the smallest element with cumulative > q.
            res.append(cumGcd[lo][0])
        return res