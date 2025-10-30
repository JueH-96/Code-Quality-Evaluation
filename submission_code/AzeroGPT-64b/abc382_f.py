import sys
input = sys.stdin.buffer.readline

from bisect import bisect_right

H, W, N = map(int, input().split())

class SegTree:
    def __init__(self, size):
        self.size = 2 ** (size - 1).bit_length()
        self.tree = [0] * (2 * self.size)

    def replace(self, index, value):
        index += self.size
        self.tree[index] = value
        while index > 0:
            index //= 2
            self.tree[index] = min(self.tree[index * 2], self.tree[index * 2 + 1])

    def min_right(self, left, max_value):
        left += self.size
        min_val = max_value + 1
        while left < 2 * self.size:
            if left % 2 == 0:
                left += 1
            else:
                if min_val > self.tree[left]:
                    min_val = self.tree[left]
                    while left < self.size:
                        left *= 2
                        if min_val > self.tree[left]:
                            min_val = self.tree[left]
                        if min_val > self.tree[left + 1]:
                            min_val = self.tree[left + 1]
                            left += 1
                    return left - self.size
                left += 1

bars = []
for _ in range(N):
    r, c, l = map(int, input().split())
    r -= 1
    c -= 1
    bars.append((r, c, c + l))

tops = [None] * W
for i, _, c, c_l in enumerate(bars):
    for j in range(c, c_l):
        if tops[j] is None:
            tops[j] = i

segtree = SegTree(H + N)
for i, c, _ in sorted((-ci, c, i) for i, _, ci, _ in enumerate(bars)):
    segtree.replace(ci + H - 1, i)

for h in range(H - 1, -1, -1):
    bottom = segtree.tree[1]
    for w in range(W):
        if tops[w] is None or segtree.tree[tops[w] + segtree.size] > h:
            segtree.replace(h, h)

        if segtree.min_right(h, bottom) == h:
            break

    else:
        break

ans = []
for _, ci, ci_l in bars:
    ans.append(segtree.tree[ci + H])

print(*sorted(a + 1 for a in ans), sep = '
')