from typing import List

class SegmentTree:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        size = 1
        while size < n:
            size <<= 1
        self.size = size
        # segment tree arrays
        self.minv = [0] * (2 * size)
        self.maxv = [0] * (2 * size)
        self.lazy = [0] * (2 * size)
        # leaf count array (for counting)
        self.cnt = [0] * (2 * size)
        for i in range(size, size + n):
            self.cnt[i] = 1
        for i in range(size - 1, 0, -1):
            self.cnt[i] = self.cnt[2*i] + self.cnt[2*i+1]

    def _apply(self, idx, v):
        # apply lazy add v to node idx
        self.minv[idx] += v
        self.maxv[idx] += v
        self.lazy[idx] += v

    def _push(self, idx):
        # push down lazy to children
        v = self.lazy[idx]
        if v != 0:
            self._apply(idx*2, v)
            self._apply(idx*2+1, v)
            self.lazy[idx] = 0

    def _pull(self, idx):
        # recalc from children
        self.minv[idx] = min(self.minv[2*idx], self.minv[2*idx+1])
        self.maxv[idx] = max(self.maxv[2*idx], self.maxv[2*idx+1])

    def range_add(self, l, r, v, idx=1, left=0, right=None):
        # add v to [l..r]
        if right is None:
            right = self.size - 1
        if r < left or right < l:
            return
        if l <= left and right <= r:
            self._apply(idx, v)
            return
        self._push(idx)
        mid = (left + right) // 2
        self.range_add(l, r, v, idx*2, left, mid)
        self.range_add(l, r, v, idx*2+1, mid+1, right)
        self._pull(idx)

    def count_le(self, l, r, idx=1, left=0, right=None):
        # count how many in [l..r] have value <= k
        if right is None:
            right = self.size - 1
        if r < left or right < l:
            return 0
        if l <= left and right <= r:
            # whole node in range
            if self.maxv[idx] <= self.k:
                return self.cnt[idx]
            if self.minv[idx] > self.k:
                return 0
        # partial or mixed
        if left == right:
            # leaf
            return 1 if self.minv[idx] <= self.k else 0
        self._push(idx)
        mid = (left + right) // 2
        return (self.count_le(l, r, idx*2, left, mid) +
                self.count_le(l, r, idx*2+1, mid+1, right))


class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Segment tree to maintain cost[L] for subarrays starting at L ending at current r
        st = SegmentTree(n, k)
        # Blocks: list of dicts with keys l, r, m (range [l..r], M value)
        # They partition [0..r-1] in 0-based L
        blocks = []
        ans = 0
        for r in range(n):
            x = nums[r]
            # merge blocks whose M <= x
            merge_l = r
            while blocks and blocks[-1]['m'] <= x:
                b = blocks.pop()
                merge_l = b['l']
            # now remaining blocks have M > x; for them add (M - x)
            # we iterate all remaining blocks
            for b in blocks:
                # b covers [b.l..b.r], 0-based
                st.range_add(b['l'], b['r'], b['m'] - x)
            # push new merged block [merge_l..r] with M = x
            blocks.append({'l': merge_l, 'r': r, 'm': x})
            # the new position r itself has cost 0 already by default
            # count how many L in [0..r] have cost[L] <= k
            ans += st.count_le(0, r)
        return ans