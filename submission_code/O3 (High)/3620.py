from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        """
        Each element nums[i] can be moved anywhere inside the interval
        [nums[i] - k , nums[i] + k].  We want to choose, for every index,
        a final value so that the amount of different values that appear
        is maximised.

        The problem can be re-stated as follows:
           – We have n identical length intervals on the integer line
             (length = 2*k + 1, or 1 when k == 0).
           – From every interval we must pick one integer that lies in it.
           – Two different indices are allowed to pick the same integer,
             but that does not increase the number of distinct values.
           – We want to maximise how many *different* integers are picked.

        Greedy idea (classic for intervals and discrete points):
           Sort the intervals by their right end (R) in non-decreasing order.
           Keep ‘next_free’, the smallest integer that has not been used yet.
           For the current interval [L, R]:
                 x = max(L, next_free)        # first still free integer ≥ L
                 if x ≤ R:
                     use x
                     next_free = x + 1
                     answer += 1
                 else:
                     # all integers of this interval are already taken,
                     # therefore we can only duplicate –  no new
                     # distinct value is possible from this index.
           Because we always occupy the earliest still free position,
           no later interval is ever blocked while an earlier position
           would still be usable for it, so this greedy is optimal and
           yields the maximum possible number of distinct values.

        Complexity:
            – building the intervals:  O(n)
            – sorting them:            O(n log n)
            – greedy sweep:            O(n)
            – memory usage:            O(1) apart from the list of intervals
        """
        n = len(nums)
        if k == 0:
            # When k == 0 every interval is a single point,
            # so the best we can do is the amount of different numbers
            # already present.
            return len(set(nums))

        # Build intervals [L, R] (inclusive) and sort by R
        intervals = [(v - k, v + k) for v in nums]
        intervals.sort(key=lambda p: p[1])           # sort by right end

        next_free = -10**20                          # something very small
        distinct = 0

        for L, R in intervals:
            candidate = max(L, next_free)            # first still free ≥ L
            if candidate <= R:                       # we can create a new value
                distinct += 1
                next_free = candidate + 1            # smallest still free now
            # else: whole interval already occupied -> only duplicates possible

        return distinct