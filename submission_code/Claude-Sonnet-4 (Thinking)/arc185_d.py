def solve():
    N, M = map(int, input().split())
    MOD = 998244353
    result = (N * M * (N + M + 1)) % MOD
    print(result)

solve()