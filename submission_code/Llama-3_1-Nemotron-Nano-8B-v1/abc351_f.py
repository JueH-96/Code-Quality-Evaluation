import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

n = int(input())
A = list(map(int, input().split()))

sorted_unique = sorted(set(A))
size = len(sorted_unique)
count_tree = FenwickTree(size)
sum_tree = FenwickTree(size)

total = 0

for x in A:
    r = bisect.bisect_left(sorted_unique, x) + 1  # Convert to 1-based index
    cnt = count_tree.query(r)
    s = sum_tree.query(r)
    total += x * cnt - s
    count_tree.update(r, 1)
    sum_tree.update(r, x)

print(total)