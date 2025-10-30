def mod_inverse(a, p):
    return pow(a, p - 2, p)

def expected_salary(N, A):
    MOD = 998244353
    E = [0] * (N + 1)
    
    for i in range(1, N + 1):
        E[i] = (A[i - 1] + sum(E[j] for j in range(1, i))) * mod_inverse(i, MOD) % MOD

    total_expectation = sum(E[i] for i in range(1, N + 1)) * mod_inverse(N, MOD) % MOD
    return total_expectation

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
A = list(map(int, data[1:N + 1]))

result = expected_salary(N, A)
print(result)