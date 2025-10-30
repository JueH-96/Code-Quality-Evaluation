import bisect

INF = 10**18

class SegmentTree:
    def __init__(self, n: int):
        self.n = n
        self.min_tree = [INF] * (4 * n)
        self.max_tree = [-INF] * (4 * n)
    
    def update(self, idx, l, r, pos, val):
        if l == r:
            self.min_tree[idx] = val
            self.max_tree[idx] = val
            return
        mid = (l + r) // 2
        if pos <= mid:
            self.update(idx*2+1, l, mid, pos, val)
        else:
            self.update(idx*2+2, mid+1, r, pos, val)
        self.min_tree[idx] = min(self.min_tree[idx*2+1], self.min_tree[idx*2+2])
        self.max_tree[idx] = max(self.max_tree[idx*2+1], self.max_tree[idx*2+2])
    
    def query_min(self, idx, l, r, ql, qr):
        if ql > r or qr < l:
            return INF
        if ql <= l and r <= qr:
            return self.min_tree[idx]
        mid = (l + r) // 2
        left_res = INF
        right_res = INF
        if ql <= mid:
            left_res = self.query_min(idx*2+1, l, mid, ql, qr)
        if qr > mid:
            right_res = self.query_min(idx*2+2, mid+1, r, ql, qr)
        return min(left_res, right_res)
    
    def query_max(self, idx, l, r, ql, qr):
        if ql > r or qr < l:
            return -INF
        if ql <= l and r <= qr:
            return self.max_tree[idx]
        mid = (l + r) // 2
        left_res = -INF
        right_res = -INF
        if ql <= mid:
            left_res = self.query_max(idx*2+1, l, mid, ql, qr)
        if qr > mid:
            right_res = self.query_max(idx*2+2, mid+1, r, ql, qr)
        return max(left_res, right_res)

class Solution:
    def minAbsoluteDifference(self, nums, x):
        n = len(nums)
        if x == 0:
            return 0
        comp = sorted(set(nums))
        m = len(comp)
        ans = INF
        
        seg = SegmentTree(m)
        for i in range(n):
            if i >= x:
                pos = bisect.bisect_left(comp, nums[i-x])
                seg.update(0, 0, m-1, pos, comp[pos])
            j = bisect.bisect_right(comp, nums[i]) - 1
            if j >= 0:
                max_val = seg.query_max(0, 0, m-1, 0, j)
                if max_val != -INF:
                    ans = min(ans, nums[i] - max_val)
            j2 = bisect.bisect_left(comp, nums[i])
            if j2 < m:
                min_val = seg.query_min(0, 0, m-1, j2, m-1)
                if min_val != INF:
                    ans = min(ans, min_val - nums[i])
        
        seg = SegmentTree(m)
        for i in range(n-1, -1, -1):
            if i + x < n:
                pos = bisect.bisect_left(comp, nums[i+x])
                seg.update(0, 0, m-1, pos, comp[pos])
            j = bisect.bisect_right(comp, nums[i]) - 1
            if j >= 0:
                max_val = seg.query_max(0, 0, m-1, 0, j)
                if max_val != -INF:
                    ans = min(ans, nums[i] - max_val)
            j2 = bisect.bisect_left(comp, nums[i])
            if j2 < m:
                min_val = seg.query_min(0, 0, m-1, j2, m-1)
                if min_val != INF:
                    ans = min(ans, min_val - nums[i])
        
        return ans