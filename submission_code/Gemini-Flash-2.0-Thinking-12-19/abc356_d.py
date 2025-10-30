def solve():
    n, m = map(int, input().split())
    mod = 998244353
    total_sum = 0
    for i in range(60):
        if (m >> i) & 1:
            q = (n + 1) // (1 << (i + 1))
            r = (n + 1) % (1 << (i + 1))
            count_i = q * (1 << i) + max(0, r - (1 << i))
            total_sum = (total_sum + count_i) % mod
    print(total_sum)

solve()