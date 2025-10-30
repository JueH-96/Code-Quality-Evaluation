MOD = 998244353

N, K = map(int, input().split())

if N == 1:
    print(1)
    exit()

inv2 = (MOD + 1) // 2  # 499122177

term1 = (N - 1) * inv2 % MOD
a = (N - 2) * pow(N, MOD - 2, MOD) % MOD
a_power = pow(a, K, MOD)
term1 = term1 * a_power % MOD

term2 = (N + 1) * inv2 % MOD

result = (term2 - term1) % MOD
print(result)