from typing import List
import bisect

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Find the maximum number in nums to set the frequency array range.
        m = max(nums)
        freq = [0] * (m + 1)
        for num in nums:
            freq[num] += 1

        # exact[g] will store the number of pairs whose gcd is exactly g.
        exact = [0] * (m + 1)
        
        # For each potential gcd value from m down to 1,
        # count the number of numbers that are multiples of g.
        # Then compute how many pairs can be formed among those numbers.
        # Subtract pairs that have a gcd greater than g (which are computed already).
        for g in range(m, 0, -1):
            countMultiples = 0
            # Sum frequencies for all multiples of g.
            for k in range(g, m + 1, g):
                countMultiples += freq[k]
            pairs = countMultiples * (countMultiples - 1) // 2
            # Subtract pairs where gcd is a multiple of g greater than g.
            for k in range(2 * g, m + 1, g):
                pairs -= exact[k]
            exact[g] = pairs

        # Build a sorted list of (g, count) where count > 0.
        # Since gcd values can only be between 1 and m, and in sorted order,
        # the sorted gcdPairs (in ascending order) is like:
        # [g repeated exact[g] times] for each g in increasing order.
        gcd_list = []  # List of (g, cumulative frequency)
        cumulative = 0
        for g in range(1, m + 1):
            if exact[g] > 0:
                cumulative += exact[g]
                gcd_list.append((g, cumulative))
        
        # For each query, we perform a binary search on the cumulative counts list.
        # Since queries[i] is a 0-indexed position in the sorted gcdPairs list,
        # we want to find the first element (g, cumulative count) for which
        # cumulative count > query.
        res = []
        # We'll extract the cumulative counts into a separate list for binary search.
        cum_vals = [cp for (_, cp) in gcd_list]
        for q in queries:
            # bisect_right returns the insertion point after any existing entries equal to q.
            # This is exactly what we need because the cumulative count indicates the last index
            # in that bucket (0-indexed). So the first bucket with cum_val > q is the answer.
            idx = bisect.bisect_right(cum_vals, q)
            res.append(gcd_list[idx][0])
        return res