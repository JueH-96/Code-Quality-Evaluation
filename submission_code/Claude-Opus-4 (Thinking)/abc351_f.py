# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Coordinate compression
vals = sorted(set(a))
coord = {v: i + 1 for i, v in enumerate(vals)}  # 1-indexed for Fenwick tree

class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree_count = [0] * (n + 1)
        self.tree_sum = [0] * (n + 1)
    
    def update(self, i, val):
        while i <= self.n:
            self.tree_count[i] += 1
            self.tree_sum[i] += val
            i += i & (-i)
    
    def query_count(self, i):
        s = 0
        while i > 0:
            s += self.tree_count[i]
            i -= i & (-i)
        return s
    
    def query_sum(self, i):
        s = 0
        while i > 0:
            s += self.tree_sum[i]
            i -= i & (-i)
        return s

ft = FenwickTree(len(vals))
result = 0

for j in range(n):
    # Find count and sum of elements < a[j]
    idx = coord[a[j]] - 1  # Elements strictly less than a[j]
    if idx > 0:
        count = ft.query_count(idx)
        sum_val = ft.query_sum(idx)
        result += a[j] * count - sum_val
    
    # Insert a[j]
    ft.update(coord[a[j]], a[j])

print(result)