import sys
input = sys.stdin.readline

class BIT:
    def __init__(self, n):
        self.size = n
        self.tree = [0] * (n + 1)
 
    def sum(self, i):
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s
 
    def add(self, i, x):
        while i <= self.size:
            self.tree[i] += x
            i += i & -i

    def lower_bound(self, w):
        if w <= 0:
            return 0
        x = 0
        k = 1 << ((self.size).bit_length() - 1)
        while k > 0:
            if x + k <= self.size and self.tree[x + k] < w:
                w -= self.tree[x + k]
                x += k
            k //= 2
        return x + 1

n = int(input())
h = list(map(int, input().split()))
bit = BIT(n)
ans = [0] * n
for i, x in enumerate(h[::-1]):
    i = n - i - 1
    x += bit.sum(i + 1)
    ans[i] = bit.lower_bound(x)
    bit.add(i + 1, 1)
print(*ans)