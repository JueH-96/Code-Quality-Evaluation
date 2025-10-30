MOD = 998244353

N = int(input())
A = list(map(int, input().split()))

sum_A = [0] * (N + 1)
for i in range(N-1, -1, -1):
    sum_A[i] = (sum_A[i+1] + A[i]) % MOD

inv_N = pow(N, MOD-2, MOD)
sum_E = 0
E = [0] * (N + 1)

for x in range(N, -1, -1):
    E[x] = (sum_A[x] + sum_E) * inv_N % MOD
    sum_E = (sum_E + E[x]) % MOD

print(E[0] % MOD)