from bisect import bisect_left
from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        if not intervals:
            return []
        
        k_max = 4                          # we can take at most 4 intervals
        n      = len(intervals)

        # ------------------------------------------------------------------
        # 1.  sort the intervals by their right end, keep original indices
        # ------------------------------------------------------------------
        data = []                          # (right, left, weight, original_index)
        for idx, (l, r, w) in enumerate(intervals):
            data.append((r, l, w, idx))
        data.sort()

        # build helper arrays  (1-based for convenience, position 0 is sentinel)
        rights  = [0]                      # right end
        lefts   = [0]                      # left  start
        weights = [0]                      # weight
        idxs    = [0]                      # original indices (0-based)
        for r, l, w, idx in data:
            rights.append(r)
            lefts.append(l)
            weights.append(w)
            idxs.append(idx)

        n = len(rights) - 1                # real number of intervals after sentinel

        # ------------------------------------------------------------------
        # 2.  prev[i]  = largest j < i with rights[j] < lefts[i]   (no overlap)
        # ------------------------------------------------------------------
        prev = [0] * (n + 1)
        for i in range(1, n + 1):
            prev[i] = bisect_left(rights, lefts[i]) - 1     # rights[prev[i]] < lefts[i]

        # ------------------------------------------------------------------
        # 3.  DP :  dp[m][i]  best weight & indices using exactly m intervals
        #           among first i intervals (by end order)
        # ------------------------------------------------------------------
        NEG = -1                                            # impossible sentinel
        dp_w   = [[NEG] * (n + 1) for _ in range(k_max + 1)]
        dp_lst = [[[]  for _ in range(n + 1)] for _ in range(k_max + 1)]

        # zero intervals ⇒ weight 0, empty list
        for i in range(n + 1):
            dp_w[0][i]   = 0
            dp_lst[0][i] = []

        for i in range(1, n + 1):
            # first, inherit "do not take interval i"
            for m in range(k_max + 1):
                dp_w[m][i]   = dp_w[m][i - 1]
                dp_lst[m][i] = dp_lst[m][i - 1]

            # second, try to take interval i
            for m in range(1, k_max + 1):
                if dp_w[m - 1][prev[i]] == NEG:
                    continue

                cand_w   = dp_w[m - 1][prev[i]] + weights[i]
                cand_lst = dp_lst[m - 1][prev[i]] + [idxs[i]]
                cand_lst.sort()                               # keep indices sorted

                if cand_w > dp_w[m][i]:
                    dp_w[m][i]   = cand_w
                    dp_lst[m][i] = cand_lst
                elif cand_w == dp_w[m][i]:
                    # tie – pick lexicographically smaller list
                    if not dp_lst[m][i] or cand_lst < dp_lst[m][i]:
                        dp_lst[m][i] = cand_lst

        # ------------------------------------------------------------------
        # 4.  pick best among m = 1 … 4 (could also be 0 if every weight was ≤ 0)
        # ------------------------------------------------------------------
        best_w   = NEG
        best_lst = []
        for m in range(1, k_max + 1):
            w   = dp_w[m][n]
            lst = dp_lst[m][n]
            if w > best_w:
                best_w, best_lst = w, lst
            elif w == best_w and lst and lst < best_lst:
                best_lst = lst

        return best_lst