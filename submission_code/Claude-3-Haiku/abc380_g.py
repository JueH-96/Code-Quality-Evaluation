import sys
from math import factorial

MOD = 998244353

def expected_inversion_number(N, K, P):
    # Calculate the initial inversion number
    inversion_count = 0
    for i in range(N):
        for j in range(i+1, N):
            if P[i] > P[j]:
                inversion_count += 1
    
    # Calculate the expected change in inversion number
    expected_change = 0
    for i in range(N-K+1):
        sub_perm = P[i:i+K]
        sub_inversion_count = 0
        for j in range(K):
            for k in range(j+1, K):
                if sub_perm[j] > sub_perm[k]:
                    sub_inversion_count += 1
        expected_change += sub_inversion_count * factorial(K-1) / factorial(K)**2
    
    # Calculate the final expected inversion number
    expected_inversion_number = (inversion_count + expected_change) % MOD
    return int(expected_inversion_number)

# Read input
N, K = map(int, input().split())
P = list(map(int, input().split()))

# Solve the problem
result = expected_inversion_number(N, K, P)
print(result)