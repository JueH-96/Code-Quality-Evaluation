from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 1_000_000_007
        NEG_INF = -10**18           # something surely smaller than any possible answer
        n = len(nums)

        # --- segment tree -----------------------------------------------------
        size = 1
        while size < n:
            size <<= 1

        # each node keeps 2 x 2 matrix
        # dp[a][b] = best sum inside the segment
        #            a : whether the first element of the segment is taken   (0 / 1)
        #            b : whether the last  element of the segment is taken   (0 / 1)
        tree = [[[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]] for _ in range(size << 1)]

        def leaf_matrix(val: int):
            """matrix for a segment of length 1"""
            m = [[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]
            m[0][0] = 0              # choose nothing
            m[1][1] = val            # choose the element itself
            return m

        def combine(A, B):
            """merge two matrices that represent adjacent segments"""
            C = [[NEG_INF, NEG_INF], [NEG_INF, NEG_INF]]
            for hl in (0, 1):
                for tl in (0, 1):
                    if A[hl][tl] == NEG_INF:
                        continue
                    for hr in (0, 1):
                        if tl and hr:        # cannot take two adjacent items
                            continue
                        for tr in (0, 1):
                            if B[hr][tr] == NEG_INF:
                                continue
                            new_h, new_t = hl, tr
                            val = A[hl][tl] + B[hr][tr]
                            if val > C[new_h][new_t]:
                                C[new_h][new_t] = val
            return C

        # build leaves
        for i in range(n):
            tree[size + i] = leaf_matrix(nums[i])
        for i in range(size - 1, 0, -1):
            tree[i] = combine(tree[i << 1], tree[i << 1 | 1])

        def point_update(pos: int, val: int):
            """set nums[pos] = val"""
            idx = size + pos
            tree[idx] = leaf_matrix(val)
            idx >>= 1
            while idx:
                tree[idx] = combine(tree[idx << 1], tree[idx << 1 | 1])
                idx >>= 1

        # ----------------------------------------------------------------------

        total = 0
        for pos, x in queries:
            point_update(pos, x)
            best = max(max(row) for row in tree[1])  # root represents whole array
            if best < 0:
                best = 0
            total = (total + best) % MOD

        return total