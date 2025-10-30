mod = 998244353

class ModInt:
    def __init__(self, x):
        self.x = x % mod

    def __str__(self):
        return str(self.x)

    __repr__ = __str__

    def __add__(self, other):
        return (
            ModInt(self.x + other.x)
            if isinstance(other, ModInt)
            else ModInt(self.x + other)
        )

    def __sub__(self, other):
        return (
            ModInt(self.x - other.x)
            if isinstance(other, ModInt)
            else ModInt(self.x - other)
        )

    def __mul__(self, other):
        return (
            ModInt(self.x * other.x)
            if isinstance(other, ModInt)
            else ModInt(self.x * other)
        )

    def __truediv__(self, other):
        return (
            ModInt(self.x * pow(other.x, mod - 2, mod))
            if isinstance(other, ModInt)
            else ModInt(self.x * pow(other, mod - 2, mod))
        )

    def __pow__(self, n):
        if n == 0:
            return ModInt(1)
        elif n % 2 == 1:
            return self * (self ** (n - 1))
        else:
            a = self ** (n // 2)
            return a * a

from math import ceil, log2

class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.bf = [0] * (self.n + 1)

    def sum(self, i):
        s = 0
        while i > 0:
            s += self.bf[i]
            i -= i & -i
        return ModInt(s)

    def add(self, i, x):
        while i <= self.n:
            self.bf[i] += x
            i += i & -i

    def cumsum(self):
        return [self.sum(i) for i in range(1, self.n + 1)]

N, K = map(int, input().split())
A = list(map(int, input().split()))

cumsum = [0]
for a in A:
    cumsum.append(cumsum[-1] + a)

from bisect import bisect_right

bf = FenwickTree(N + 2)

ans = ModInt(1 << (N - 1))

for i in range(N + 1):
    val = cumsum[i] - K
    pt = bisect_right(cumsum, val) - 1
    if pt < 0:
        continue
    res = bf.sum(pt + 1)
    ans -= res * 2
    bf.add(i + 1, 1)

ans += ModInt(1) * (bf.sum(N + 1) == 0)

print(ans)