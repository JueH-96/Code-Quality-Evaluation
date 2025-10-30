def solve():
    n, k = map(int, input().split())
    mod = 998244353

    def mod_inv(x, p):
        return pow(x, p - 2, p)

    expected_pos = 1
    for _ in range(k):
        expected_pos = (expected_pos * (n * n - 2 * n + 2) % mod * mod_inv(n * n, mod) % mod + (n + 1) * (n - expected_pos) % mod * mod_inv(n * n, mod) % mod) % mod

    print(int(expected_pos))

solve()