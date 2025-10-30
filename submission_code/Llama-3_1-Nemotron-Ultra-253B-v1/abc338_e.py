import bisect

class BIT:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, idx, delta):
        idx += 1  # 1-based index
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx
    
    def query(self, idx):
        idx += 1  # 1-based index
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

n = int(input())
chords = []
es = []

for _ in range(n):
    a, b = map(int, input().split())
    if (b - a) % (2 * n) < n:
        s, e = a, b
    else:
        s, e = b, a
    if s > e:
        e += 2 * n
    chords.append((s, e))
    es.append(e)

chords.sort(key=lambda x: x[0])
sorted_e = sorted(es)
bit = BIT(len(sorted_e))

for s, e in chords:
    left = s
    right = e - 1
    l = bisect.bisect_left(sorted_e, left)
    r = bisect.bisect_right(sorted_e, right) - 1
    if l <= r:
        cnt = bit.query(r)
        if l > 0:
            cnt -= bit.query(l - 1)
        if cnt > 0:
            print("Yes")
            exit()
    idx = bisect.bisect_left(sorted_e, e)
    bit.update(idx, 1)

print("No")