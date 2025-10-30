def solve(N, M):
    MOD = 998244353
    if N == 2:
        return (M * (M - 1)) % MOD
    else:
        # First person has M choices, second person has M-1 choices
        total_ways = M * (M - 1) % MOD
        # Each subsequent person has M-1 choices
        total_ways = total_ways * pow(M - 1, N - 2, MOD) % MOD
        return total_ways

N, M = map(int, input().split())
print(solve(N, M))