import sys
data = sys.stdin.read().split()
N = int(data[0])
M = int(data[1])
MOD = 998244353
first_term = pow(M - 1, N, MOD)
sign_part = pow(-1, N, MOD)
second_term = ((sign_part * (M - 1)) % MOD)
result = ((first_term + second_term) % MOD)
print(result)