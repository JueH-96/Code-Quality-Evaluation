# YOUR CODE HERE
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree_sum = [0] * (n + 1)
        self.tree_count = [0] * (n + 1)
    
    def update(self, i, val):
        while i <= self.n:
            self.tree_sum[i] += val
            self.tree_count[i] += 1
            i += i & (-i)
    
    def query_sum(self, i):
        s = 0
        while i > 0:
            s += self.tree_sum[i]
            i -= i & (-i)
        return s
    
    def query_count(self, i):
        s = 0
        while i > 0:
            s += self.tree_count[i]
            i -= i & (-i)
        return s

n = int(input())
a = list(map(int, input().split()))

# Coordinate compression
sorted_vals = sorted(set(a))
val_to_idx = {val: i + 1 for i, val in enumerate(sorted_vals)}

ft = FenwickTree(len(sorted_vals))
total = 0

for j in range(n):
    # Find contribution of a[j]
    # We want sum of max(a[j] - a[i], 0) for all i < j
    # This equals a[j] * count(a[i] < a[j]) - sum(a[i] where a[i] < a[j])
    
    idx = val_to_idx[a[j]]
    if idx > 1:
        count_smaller = ft.query_count(idx - 1)
        sum_smaller = ft.query_sum(idx - 1)
        total += a[j] * count_smaller - sum_smaller
    
    # Add current element to fenwick tree
    ft.update(idx, a[j])

print(total)