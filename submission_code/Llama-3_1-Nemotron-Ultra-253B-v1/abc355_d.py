import bisect

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 1)  # 1-based indexing

    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

n = int(input())
intervals = []
all_values = []
for _ in range(n):
    l, r = map(int, input().split())
    intervals.append((l, r))
    all_values.append(l)
    all_values.append(r)

# Compress coordinates
sorted_unique = sorted(list(set(all_values)))
m = len(sorted_unique)

# Sort intervals by l
intervals.sort(key=lambda x: x[0])

# Initialize Fenwick Tree
ft = FenwickTree(m)
ans = 0
sum_total = 0

for l, r in intervals:
    # Query for l
    idx = bisect.bisect_left(sorted_unique, l)
    compressed_idx = idx + 1
    count = sum_total - ft.query(compressed_idx - 1)
    ans += count

    # Insert r
    r_idx = bisect.bisect_left(sorted_unique, r)
    r_compressed = r_idx + 1
    ft.update(r_compressed, 1)
    sum_total += 1

print(ans)