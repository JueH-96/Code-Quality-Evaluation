from typing import List
from collections import defaultdict

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        """
        For every point (x, y) we look for all earlier points (x', y') that
        satisfy (x ^ x') + (y ^ y') == k.

        Because k ≤ 100, there are at most k+1 possible decompositions:
               (dx, dy) with dx + dy = k  and  0 ≤ dx ≤ k
        and once (dx, dy) is fixed the required earlier point is
               (x' , y') = (x ^ dx , y ^ dy).

        We keep a dictionary that stores how many times each point has been
        seen so far, query it for every decomposition, then add the current
        point to the dictionary.

        Time complexity :  O(n · k)  ( ≤ 5 000 000 operations for the
                                      given limits)
        Space complexity:  O(n)
        """
        seen = defaultdict(int)      # (x, y) -> occurrences so far
        ans = 0

        for x, y in coordinates:
            for dx in range(k + 1):
                dy = k - dx
                px = x ^ dx          # required x-coordinate of the earlier point
                py = y ^ dy          # required y-coordinate of the earlier point
                ans += seen[(px, py)]
            seen[(x, y)] += 1

        return ans