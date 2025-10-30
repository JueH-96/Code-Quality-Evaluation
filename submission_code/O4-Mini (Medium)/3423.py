from typing import List

class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # Special case n == 0 (though constraints say n >= 1)
        if n == 0:
            return 0
        # Define negative infinity for max-plus semiring
        NEG_INF = -10**30
        # Identity matrix in max-plus: [[0, -inf], [-inf, 0]]
        I = (0, NEG_INF, NEG_INF, 0)
        # Leaf matrix constructor for value v: [[0, 0], [v, -inf]]
        def make_leaf(v: int):
            return (0, 0, v, NEG_INF)
        # Merge two matrices L, R to compute R @ L in max-plus
        def merge(L, R):
            a, b, c, d = L   # L = [a b; c d]
            e, f, g, h = R   # R = [e f; g h]
            # result = R @ L
            r00 = e + a
            if f + c > r00: r00 = f + c
            r01 = e + b
            if f + d > r01: r01 = f + d
            r10 = g + a
            if h + c > r10: r10 = h + c
            r11 = g + b
            if h + d > r11: r11 = h + d
            return (r00, r01, r10, r11)
        # Build tree size = next power of two >= n
        size = 1
        while size < n:
            size <<= 1
        # Initialize segment tree with identity matrices
        tree = [I] * (2 * size)
        # Set leaves
        for i in range(n):
            tree[size + i] = make_leaf(nums[i])
        # Build internal nodes
        for i in range(size - 1, 0, -1):
            L = tree[2*i]
            R = tree[2*i + 1]
            tree[i] = merge(L, R)
        # Process queries
        total = 0
        for pos, x in queries:
            # Update leaf
            idx = size + pos
            tree[idx] = make_leaf(x)
            # Recompute up to root
            idx //= 2
            while idx >= 1:
                L = tree[2*idx]
                R = tree[2*idx + 1]
                tree[idx] = merge(L, R)
                idx //= 2
            # After update, root is tree[1]
            m00, m01, m10, m11 = tree[1]
            # dp_end = [dp0, dp1] = [m00, m10]
            ans = m00 if m00 >= m10 else m10
            total = (total + ans) % MOD
        return total