import bisect

N = int(input())
intervals = []
for i in range(N):
    l, r = map(int, input().split())
    intervals.append((l, r))

# Sort intervals by left endpoint
intervals.sort()

# Coordinate compression for right endpoints
rights = [r for l, r in intervals]
sorted_rights = sorted(set(rights))
right_to_idx = {v: i for i, v in enumerate(sorted_rights)}

# Fenwick Tree (Binary Indexed Tree)
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
    
    def query(self, i):
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= i & (-i)
        return res
    
    def range_query(self, l, r):
        if l > r:
            return 0
        return self.query(r) - self.query(l - 1)

ft = FenwickTree(len(sorted_rights))

count = 0
for l, r in intervals:
    # Find how many previous intervals have right endpoint >= l
    idx = bisect.bisect_left(sorted_rights, l)
    if idx < len(sorted_rights):
        count += ft.range_query(idx + 1, len(sorted_rights))
    
    # Add current interval's right endpoint to the data structure
    r_idx = right_to_idx[r] + 1  # 1-indexed for Fenwick tree
    ft.update(r_idx, 1)

print(count)