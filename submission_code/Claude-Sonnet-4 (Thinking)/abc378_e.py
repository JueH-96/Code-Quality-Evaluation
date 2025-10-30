class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)
    
    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
    
    def query(self, i):
        result = 0
        while i > 0:
            result += self.tree[i]
            i -= i & (-i)
        return result

n, m = map(int, input().split())
a = list(map(int, input().split()))

# Compute prefix sums modulo m
q = [0]
prefix = 0
for i in range(n):
    prefix = (prefix + a[i]) % m
    q.append(prefix)

# Compute the first term
first_term = sum((2 * k - n) * q[k] for k in range(n + 1))

# Compute the number of inversions using Fenwick tree
ft = FenwickTree(m)
inversions = 0

for j in range(n + 1):
    qj = q[j]
    
    # Count how many i < j have Q[i] > Q[j]
    if qj + 1 < m:
        inversions += ft.query(m) - ft.query(qj + 1)
    
    # Update Fenwick tree
    ft.update(qj + 1, 1)

result = first_term + m * inversions
print(result)