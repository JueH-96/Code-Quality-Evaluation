class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        sseg = SegmentTree(n)
        
        for i, x in enumerate(nums):
            sseg.add(i, i, x)

        queries = sorted(
            (r, l, i) for i, (l, r) in enumerate(queries))

        lo, lo_inv, hi = 0, 0, len(queries) - 1

        while lo <= hi:
            mid = lo + (hi - lo) // 2
            idx = queries[mid][2]

            res = self.count_zeros(sseg, queries[:mid + 1])
            if res == -1:
                lo = mid + 1
            else:
                lo_inv = max(lo_inv, mid + 1)
                hi = mid - 1

        if lo_inv == 0:
            return -1
        return lo_inv
    
    
    def count_zeros(self, sseg, queries):
        n = sseg.n
        used = [0] * n
        
        for r, l, _ in queries:
            sseg.add(l, r + 1, -1)
            used[l] += 1
            sseg.update(r + 1)

        for x in sseg.res:
            if x != 0:
                return -1
        
        return sum(x != 0 for x in used)
    
    def naive(self, nums, queries):
        if not nums:
            return 0
        sset = set([i for i, x in enumerate(nums) if x])
        added = {}
        m, n = len(queries), len(nums)
        
        res = -1

        for i in range(m):
            r, l = queries[i]
            for j in range(l, r + 1):
                if j in sset:
                    sset.remove(j)
            added[i] = len(sset)
            res = max(res, m - i - added[i])
        return res if len(sset) == 0 else -1


class SegmentTree:
    def __init__(self, n, op=min, e=0,):
        d = self.d = 1 << (n - 1).bit_length()
        self.n = n
        self.e, self.op = e, op
        self.res = self.seg = [e] * 2 * d
        self.lazy = [None] * 2 * d

    def clear(self):
        n = self.n
        for i in range(n):
            self.res[i + self.d] = self.seg[i + self.d] = 0
        for i in range(self.d - 1, 0, -1):
            self.seg[i] = self.res[i * 2]

    def build(self, a):
        n = min(len(a), self.n)
        self.clear()
        for i in range(n):
            self.seg[i + self.d] = a[i]
        for i in range(self.d - 1, 0, -1):
            self.seg[i] = self.op(self.seg[i << 1], self.seg[i << 1 | 1])

    def update(self, index):
        i = index + self.d
        self.seg[index + self.d] = self.res[index + self.d] = self.res[i] + self.res[i ^ 1]
        while i >>= 1:
            self.seg[i] = self.res[i * 2] + self.res[i * 2 + 1]

    def query(self, l, r):
        l, r = max(l, 0), min(self.n, r)
        if r - l < 0:
            return self.e

        l += self.d
        r += self.d
        e = self.e
        while l < r:
            if l & 1:
                e = self.op(self.res[l], e)
                l += 1
            if r & 1:
                r -= 1
                e = self.op(self.res[r], e)
            l >>= 1
            r >>= 1

        return e

    def range_update(self, l, r, x):
        l, r = max(l, 0), min(self.n, r)
        if not l < r or x is None:
            return

        d, n = self.d, self.n
        L = left = (l + d) >> 1
        R = right = (r + d - 1) >> 1

        def push(i):
            if not self.lazy[i] is None:
                w, self.lazy[i] = self.lazy[i], None
                self.seg[i * 2] = self.op(self.seg[i * 2], w)
                self.seg[i * 2 + 1] = self.op(self.seg[i * 2 + 1], w)
                self.res[i * 2] = self.op(self.res[i * 2], w)
                self.res[i * 2 + 1] = self.op(self.res[i * 2 + 1], w)
                self.lazy[i * 2] = self.lazy[i * 2 + 1] = self.op(self.lazy[i * 2], w) if not self.lazy[i * 2] is None else w

        while left < right:
            if left & 1:
                push(left)
                self.seg[left] = self.op(self.seg[left], x)
                self.res[left] = self.op(self.res[left], x)
                self.lazy[left] = self.op(self.lazy[left], x) if not self.lazy[left] is None else x
                left += 1

            if right & 1:
                right -= 1
                push(right)
                self.seg[right] = self.op(self.seg[right], x)
                self.res[right] = self.op(self.res[right], x)
                self.lazy[right] = self.op(self.lazy[right], x) if not self.lazy[right] is None else x

            left >>= 1
            right >>= 1

        for i in range(left, right + 1):
            if i & -i == i:
                push(i)

        i = L
        while i < d:
            self.seg[i] = self.op(self.seg[i * 2], self.seg[i * 2 + 1])
            i <<= 1

        i = R

        while i < d:
            self.seg[i] = self.op(self.seg[i * 2], self.seg[i * 2 + 1])
            i <<= 1

    def add(self, l, r, x):
        self.range_update(l, r, x)