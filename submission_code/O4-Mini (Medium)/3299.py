from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter
        freq = Counter(nums)
        ans = 1
        # Special case for x = 1, since 1^anything = 1 causes infinite loop
        if freq.get(1, 0) > ans:
            ans = freq[1]
        # Process other bases
        LIMIT = 10**9
        for x, cnt_x in freq.items():
            if x == 1:
                continue
            # build the chain x^(2^0), x^(2^1), x^(2^2), ...
            levels = []
            cur = x
            while True:
                if cur not in freq:
                    break
                levels.append(cur)
                # stop if squaring would overflow beyond LIMIT
                if cur > LIMIT // cur:
                    break
                cur = cur * cur
            # levels now holds all valid chain values for this base
            # try all possible m from 0..len(levels)-1
            # m=0 => sequence [x], length=1
            # for m>=1 => need freq[level[j]]>=2 for j<m, and freq[level[m]]>=1
            best_for_x = 1  # at least we can pick [x]
            L = len(levels)
            for m in range(L):
                ok = True
                # check double-requirement for lower levels
                for j in range(m):
                    if freq[levels[j]] < 2:
                        ok = False
                        break
                if not ok:
                    continue
                # check the top level needs at least 1
                # (always true since levels are from freq keys)
                # compute length = 2*m + 1
                length = 2 * m + 1
                if length > best_for_x:
                    best_for_x = length
            if best_for_x > ans:
                ans = best_for_x
        return ans