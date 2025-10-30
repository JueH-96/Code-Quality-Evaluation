MOD = 998244353

N, M = map(int, input().split())

part1 = pow(M-1, N, MOD)
sign = (-1) ** N
part2 = sign * (M - 1)
ans = (part1 + part2) % MOD

print(ans)