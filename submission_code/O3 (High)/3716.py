from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        # maximum possible value and absolute difference according to constraints
        MAX_VAL = 300          # 1 .. 300
        MAX_DIFF = 300         # 0 .. 299, we add 300 as the "infinite" first diff

        SZ = MAX_DIFF + 2      # size of Fenwick tree arrays (1-based)

        # One Fenwick tree (for range-maximum queries) for each value 1 .. 300
        trees = [[0] * SZ for _ in range(MAX_VAL + 1)]

        # ------------- Fenwick helpers (works with indices 1 .. SZ-1) -----------------
        # We store   idx = MAX_DIFF - diff + 1
        # so that asking for “max diff >= d” becomes a prefix query up to idx.
        def bit_query(tree, diff):
            idx = MAX_DIFF - diff + 1
            res = 0
            while idx:
                if tree[idx] > res:
                    res = tree[idx]
                idx -= idx & -idx
            return res

        def bit_update(tree, diff, val):
            idx = MAX_DIFF - diff + 1
            n = SZ
            while idx < n:
                if val > tree[idx]:
                    tree[idx] = val
                idx += idx & -idx
        # ---------------------------------------------------------------------------

        best = 1                       # answer is at least one element present
        INF_DIFF = MAX_DIFF            # 300 – used for the very first difference

        for x in nums:
            # collect all updates first (so that current element is not reused)
            pending = []

            for v in range(1, MAX_VAL + 1):            # every possible previous value
                d = abs(x - v)                         # current absolute difference
                prev_len = bit_query(trees[v], d)      # best seq ending with v, diff ≥ d
                if prev_len:
                    pending.append((d, prev_len + 1))  # extend it with current number

            # subsequence consisting of the single current element
            pending.append((INF_DIFF, 1))

            # apply all prepared updates
            t = trees[x]
            for d, length in pending:
                bit_update(t, d, length)
                if length > best:
                    best = length

        return best