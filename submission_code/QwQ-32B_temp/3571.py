import bisect

class FenwickTreeMax:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, idx, value):
        while idx <= self.n:
            if value > self.tree[idx]:
                self.tree[idx] = value
            else:
                break  # No need to proceed if the new value is not larger
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            if self.tree[idx] > res:
                res = self.tree[idx]
            idx -= idx & -idx
        return res

class SegmentTree:
    def __init__(self, size):
        self.n = size
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [0] * (2 * self.size)

    def update(self, idx, value):
        # idx is 1-based
        pos = idx + self.size - 1
        self.tree[pos] = value
        pos >>= 1
        while pos >= 1:
            new_val = max(self.tree[2 * pos], self.tree[2 * pos + 1])
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
            pos >>= 1

    def query_range(self, l, r):
        # l and r are 1-based
        res = 0
        l += self.size - 1
        r += self.size - 1
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Step 1: Sort the coordinates by x then y, and track original indices
        sorted_coords = sorted([(x, y, i) for i, (x, y) in enumerate(coordinates)], key=lambda x: (x[0], x[1]))
        
        # Find the position of the k-th element in the sorted list
        pos = -1
        for i in range(len(sorted_coords)):
            if sorted_coords[i][2] == k:
                pos = i
                break
        
        n = len(sorted_coords)
        if n == 0:
            return 0
        
        # Step 2: Assign ranks to the y's
        all_ys = [y for x, y, idx in sorted_coords]
        sorted_ys = sorted(all_ys)
        rank = {}
        for i in range(len(sorted_ys)):
            y = sorted_ys[i]
            rank[y] = i + 1  # ranks are 1-based
        max_rank = len(sorted_ys)
        
        # Step 3: Compute dp1 using Fenwick Tree
        fenwick = FenwickTreeMax(max_rank)
        dp1 = [0] * n
        for i in range(n):
            x, y, idx = sorted_coords[i]
            current_rank = rank[y]
            current_max = fenwick.query(current_rank - 1)
            dp1[i] = current_max + 1
            fenwick.update(current_rank, dp1[i])
        
        # Step 4: Compute dp2 using Segment Tree
        st = SegmentTree(max_rank)
        dp2 = [0] * n
        for i in reversed(range(n)):
            x, y, idx = sorted_coords[i]
            current_rank = rank[y]
            l = current_rank + 1
            r = max_rank
            if l > r:
                max_val = 0
            else:
                max_val = st.query_range(l, r)
            dp2[i] = max_val + 1
            st.update(current_rank, dp2[i])
        
        # Step 5: Calculate the result
        return dp1[pos] + dp2[pos] - 1