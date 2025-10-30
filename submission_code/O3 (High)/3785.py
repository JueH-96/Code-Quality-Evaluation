from typing import List

class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        """
        If the consecutive differences of two arrays are the same, the second
        array has to be the first one shifted by a fixed value `d`.

            copy[i] = original[i] + d   for every i

        For every position we therefore need

            u_i ≤ original[i] + d ≤ v_i
            └─────────────────────┘
                   bounds for d

        That gives an interval for d:
            [u_i - (original[i]-original[0]),  v_i - (original[i]-original[0])]

        The intersection of all these intervals is the set of all valid shifts.
        Its size (number of integers it contains) is the answer.
        """
        base = original[0]

        # current intersection [low, high]
        low = -10**20          # sufficiently small "−∞"
        high = 10**20          # sufficiently large "+∞"

        for i, (u, v) in enumerate(bounds):
            offset = original[i] - base         # original[i] = base + offset
            low  = max(low,  u - offset)        # tighten lower bound
            high = min(high, v - offset)        # tighten upper bound
            if low > high:                      # empty intersection – impossible
                return 0

        # number of integers in [low, high]
        return high - low + 1