MOD = 998244353

N, M, K = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(M)]

class SState:
    def __init__(self):
        self.array = [0] * N
        self.shift = 0

s = SState()

for k in range(1, K + 1):
    new_shift = (s.shift + 1) % N
    current_X = (k - 1) % N + 1
    for X, Y in edges:
        a = 1 if X == current_X else 0
        idx_prev = (X - s.shift) % N
        b = s.array[idx_prev]
        contribution = (a + b) % MOD
        idx_new = (Y - new_shift) % N
        s.array[idx_new] = (s.array[idx_new] + contribution) % MOD
    s.shift = new_shift

ans = (1 + sum(s.array)) % MOD
print(ans)