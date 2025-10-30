class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        # next power-of-two >= n
        n2 = 1
        while n2 < n:
            n2 <<= 1
        # use a very small number as -infinity
        neg_inf = -10**30
        # identity matrix for max-plus semiring: [[0, -inf],[-inf, 0]]
        I = [0, neg_inf, neg_inf, 0]
        size = 2 * n2
        # segment tree array storing 2x2 matrices as flat lists [a00,a01,a10,a11]
        tree = [I[:] for _ in range(size)]
        # initialize leaves
        for i, v in enumerate(nums):
            # for a single element v: M = [[0,0],[v,-inf]]
            tree[n2 + i] = [0, 0, v, neg_inf]
        # build the tree bottom-up
        for i in range(n2 - 1, 0, -1):
            L = tree[2*i]
            R = tree[2*i + 1]
            # compute R * L in max-plus semiring
            a = R[0] + L[0]
            t = R[1] + L[2]
            if t > a: a = t
            b = R[0] + L[1]
            t = R[1] + L[3]
            if t > b: b = t
            c = R[2] + L[0]
            t = R[3] + L[2]
            if t > c: c = t
            d = R[2] + L[1]
            t = R[3] + L[3]
            if t > d: d = t
            tree[i] = [a, b, c, d]
        # process queries
        ans = 0
        mod = 10**9 + 7
        for pos, x in queries:
            # update leaf
            idx = pos + n2
            tree[idx] = [0, 0, x, neg_inf]
            # rebuild path to root
            idx //= 2
            while idx:
                L = tree[2*idx]
                R = tree[2*idx + 1]
                a = R[0] + L[0]
                t = R[1] + L[2]
                if t > a: a = t
                b = R[0] + L[1]
                t = R[1] + L[3]
                if t > b: b = t
                c = R[2] + L[0]
                t = R[3] + L[2]
                if t > c: c = t
                d = R[2] + L[1]
                t = R[3] + L[3]
                if t > d: d = t
                tree[idx] = [a, b, c, d]
                idx //= 2
            # the answer for the whole array is max(dp0, dp1)
            root = tree[1]
            cur = root[0] if root[0] > root[2] else root[2]
            ans = (ans + cur) % mod
        return ans