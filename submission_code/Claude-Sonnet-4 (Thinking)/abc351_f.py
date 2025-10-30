class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
    
    def query(self, i):
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

n = int(input())
a = list(map(int, input().split()))

# Coordinate compression
sorted_a = sorted(set(a))
coord = {v: i+1 for i, v in enumerate(sorted_a)}

# Compute left[j]: number of i < j such that A_i < A_j
left = [0] * n
ft = FenwickTree(len(sorted_a))
for j in range(n):
    idx = coord[a[j]]
    left[j] = ft.query(idx - 1)
    ft.update(idx, 1)

# Compute right[i]: number of j > i such that A_j > A_i
right = [0] * n
ft = FenwickTree(len(sorted_a))
for i in range(n-1, -1, -1):
    idx = coord[a[i]]
    right[i] = ft.query(len(sorted_a)) - ft.query(idx)
    ft.update(idx, 1)

# Compute the final sum
total = 0
for j in range(n):
    total += a[j] * left[j]
for i in range(n):
    total -= a[i] * right[i]

print(total)