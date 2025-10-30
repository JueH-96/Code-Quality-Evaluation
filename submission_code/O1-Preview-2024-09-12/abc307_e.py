# YOUR CODE HERE
MOD = 998244353
N, M = map(int, input().split())
M_minus_one = M - 1

power = pow(M_minus_one, N, MOD)

if N % 2 == 0:
    ans = (power + M_minus_one) % MOD
else:
    ans = (power - M_minus_one) % MOD
print(ans)