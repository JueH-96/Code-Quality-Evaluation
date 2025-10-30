mx = 998244353
linv = lambda x: pow(x, mx - 2, mx)

class SqrtDecomposition:
    def __init__(self, f, inv, n, m):
        self.f = f
        self.inv = inv
        self.n = n
        self.m = m
        self.K = int(n**0.5) + 1

        g = [0] * (n + m + 2)
        for i in range(n):
            g[i] = f[i]
        for i in range(n, n + m + 1):
            g[i] = inv[2]
        h = [0] * (n + m + 2)

        self.blocks = [0] * (self.K + 1)
        cur = 0
        st = self.K - 1
        for i in reversed(range(n + m + 1)):
            cur = (cur * 2 + g[i]) % mx
            if i % self.K == 0:
                self.blocks[st] = cur
                st -= 1

        st = 0
        for i in range(n, n + m + 1):
            cur = cur * inv[-i + n - m] * g[i] % mx
            if i % self.K == 0:
                self.blocks[st] = cur * self.blocks[st] % mx
                st += 1

        cur = 0
        for i in range(self.K - 1):
            cur = (cur * (self.K - i) + self.blocks[i]) % mx
        self.H = cur * self.inv(self.K) % mx

    def update(self, l, r, x):
        left = l // self.K * self.K
        right = r // self.K * self.K
        if left == right:
            for i in range(min(r + 1, self.n), left - 1, -1):
                self.blocks[left // self.K] *= self.inv[2]
                self.blocks[left // self.K] %= mx
                self.blocks[left // self.K] *= 2
                self.blocks[left // self.K] %= mx
                if i < self.n:
                    self.blocks[left // self.K] += self.f[i] != x
                    self.blocks[left // self.K] %= mx
            if left == right:
                for i in range(right, -1, -self.K):
                    self.blocks[i // self.K] *= self.inv[self.K - i % self.K]
                    self.blocks[i // self.K] %= mx
                    if i >= self.n:
                        self.blocks[i // self.K] *= 2
                        self.blocks[i // self.K] %= mx
                    self.blocks[i // self.K] *= self.K - i % self.K
                    self.blocks[i // self.K] %= mx
            return

        if l != left:
            for i in range(min(r + 1, self.n), l - 1, -1):
                self.blocks[l // self.K] *= self.inv[2]
                self.blocks[l // self.K] %= mx
                self.blocks[l // self.K] *= 2
                self.blocks[l // self.K] %= mx
                if i < self.n:
                    self.blocks[l // self.K] += self.f[i] != x
                    self.blocks[l // self.K] %= mx
            for i in range(l // self.K * self.K, -1, -self.K):
                self.blocks[i // self.K] *= self.inv[self.K - i % self.K]
                self.blocks[i // self.K] %= mx
                if i >= self.n:
                    self.blocks[i // self.K] *= 2
                    self.blocks[i // self.K] %= mx
                self.blocks[i // self.K] *= self.K - i % self.K
                self.blocks[i // self.K] %= mx

        if r != right - 1:
            for i in range(right, r + 1):
                self.f[i] = x
            for i in range(r + 1, right + self.K):
                self.blocks[right // self.K] *= self.inv[2]
                self.blocks[right // self.K] %= mx
                self.blocks[right // self.K] *= 2
                self.blocks[right // self.K] %= mx
                if i < self.n:
                    self.blocks[right // self.K] += self.f[i] != x
                    self.blocks[right // self.K] %= mx
            for i in range(right, self.n):
                self.blocks[right // self.K] *= self.inv[2]
                self.blocks[right // self.K] %= mx
                self.blocks[right // self.K] *= 2
                self.blocks[right // self.K] %= mx

        for i in range(left + self.K, right, self.K):
            self.blocks[i // self.K] = x * (self.K - i % self.K + 1) % mx
            for j in range(i + self.K - 1, self.n):
                self.blocks[i // self.K] *= self.inv[2]
                self.blocks[i // self.K] %= mx
                self.blocks[i // self.K] *= 2
                self.blocks[i // self.K] %= mx
                if j >= self.n:
                    self.blocks[i // self.K] += self.f[j] != x
                    self.blocks[i // self.K] %= mx

        self.H = 0
        for i in range(self.K - 1):
            self.H = (self.H * (self.K - i) + self.blocks[i]) % mx
        self.H = self.H * self.inv(self.K) % mx

    def valid(self, l, r):
        pos = l // self.K
        while pos * self.K < r:
            self.H = (self.H * (self.K - pos % self.K) + self.blocks[pos]) % mx
            self.H = (self.H * self.K) % mx
            pos += 1

        mask = 1
        if l % self.K != 0:
            l = l // self.K * self.K
            pos -= 1
            while l < r:
                self.H = (self.H * mask + self.blocks[pos]) % mx
                mask *= 2
                mask %= mx
                l += 1

        H = self.H * pow(2, max(0, self.n - r), mx) % mx
        return (H * pow(2, l - pos * self.K, mx)) % mx == self.blocks[pos]


n, k = list(map(int, input().split()))
s = input()
if k == 2:
    ans = 2 ** s.count("?")
    print(ans % mx)
else:
    perm = [2 ** i for i in range(n + 1 + k)]

    inv = [linv(i) for i in perm]

    good = SqrtDecomposition([1 if s[i] == "?" else 1 if s[i] == "A" else 0 for i in range(n)], inv, n, k - 1)
    ans = 1
    for i in range(n - k + 1):
        good.update(i, i + k - 1, good.valid(i, i + k))

        ans *= perm[s[i] == "?"]
        ans %= mx

    print(ans)