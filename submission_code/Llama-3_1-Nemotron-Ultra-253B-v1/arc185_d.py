MOD = 998244353

n, m = map(int, input().split())

inv3 = pow(3, MOD - 2, MOD)

result = n % MOD
result = result * m % MOD
result = result * (m + 1) % MOD
result = result * (2 * m + 1) % MOD
result = result * inv3 % MOD

print(result)