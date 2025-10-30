from typing import List
from collections import defaultdict

class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)

        # Collect indices for each distinct value
        pos = defaultdict(list)
        for idx, val in enumerate(nums):
            pos[val].append(idx)

        # Pre-compute the minimal distance for every position
        ans_per_index = [-1] * n          # default –1 (no duplicate)
        
        for indices in pos.values():
            m = len(indices)
            if m == 1:                    # value occurs only once
                continue

            # indices are already in ascending order because we scanned left→right
            for i in range(m):
                cur  = indices[i]
                nxt  = indices[(i + 1) % m]      # next occurrence (clockwise, wraps)
                prv  = indices[(i - 1) % m]      # previous occurrence (counter-clockwise, wraps)

                # clockwise distance to nxt
                d1 = nxt - cur if nxt >= cur else n - (cur - nxt)
                # counter-clockwise distance to prv
                d2 = cur - prv if cur >= prv else n - (prv - cur)

                ans_per_index[cur] = d1 if d1 < d2 else d2

        # Build answer for the required queries
        return [ans_per_index[q] for q in queries]