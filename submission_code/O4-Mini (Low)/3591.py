from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        """
        For each position i, compute the minimal cost to shift s[i] to t[i] along a circular
        alphabet. Two strategies:
          1) Move forward (next): wrap from 'z' to 'a', cost of each step depends on the
             starting letter of that step via nextCost.
          2) Move backward (previous): wrap from 'a' to 'z', cost via previousCost.
        We compute both total costs (forward and backward) and take the minimum.
        Since each character pair differs by at most 25 steps, we can afford an O(steps)
        loop per character. Overall complexity O(n * 26), which is fine for n up to 1e5.
        """
        total = 0
        # Pre-calc char-to-index for speed
        base = ord('a')
        for cs, ct in zip(s, t):
            si = ord(cs) - base
            ti = ord(ct) - base
            # forward distance in steps
            d_next = (ti - si) % 26
            cost_next = 0
            pos = si
            for _ in range(d_next):
                cost_next += nextCost[pos]
                pos = (pos + 1) % 26

            # backward distance in steps
            d_prev = (si - ti) % 26
            cost_prev = 0
            pos = si
            for _ in range(d_prev):
                cost_prev += previousCost[pos]
                pos = (pos - 1) % 26

            total += min(cost_next, cost_prev)
        return total