from typing import List

class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n // 2                    # index of the median (0-based, larger of two middles)

        less_diffs = []               # costs to raise numbers that are < k up to k
        greater_diffs = []            # costs to lower numbers that are > k down to k
        eq_cnt = 0                    # how many numbers already equal k

        for v in nums:
            if v < k:
                less_diffs.append(k - v)
            elif v > k:
                greater_diffs.append(v - k)
            else:
                eq_cnt += 1

        L = len(less_diffs)                       # how many elements < k
        G = len(greater_diffs)                    # how many elements > k

        # maximum allowed numbers strictly smaller / strictly greater than k
        max_less    = m
        max_greater = n - m - 1

        # how many elements MUST be moved to k to satisfy the two bounds
        need_from_less    = max(0, L - max_less)
        need_from_greater = max(0, G - max_greater)

        # cheapest elements to move
        less_diffs.sort()
        greater_diffs.sort()

        cost = 0
        if need_from_less:
            cost += sum(less_diffs[:need_from_less])
        if need_from_greater:
            cost += sum(greater_diffs[:need_from_greater])

        # after the compulsory moves we may still have no element equal to k
        if eq_cnt == 0 and need_from_less == 0 and need_from_greater == 0:
            # one additional element must be changed to k – pick the cheapest one
            extra = float('inf')
            if less_diffs:    # cheapest raise
                extra = min(extra, less_diffs[0])
            if greater_diffs: # cheapest decrease
                extra = min(extra, greater_diffs[0])
            cost += extra

        return cost