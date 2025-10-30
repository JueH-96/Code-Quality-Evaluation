# YOUR CODE HERE
import sys
from math import factorial

MOD = 998244353

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def count_inversions(arr):
    return sum(arr[i] > arr[j] for i in range(len(arr)) for j in range(i + 1, len(arr)))

def calculate_expected_inversions(N, K, P):
    total = 0
    factorial_K = factorial(K)
    
    for i in range(N - K + 1):
        segment = P[i:i+K]
        original_inversions = count_inversions(segment)
        
        for j in range(K):
            for k in range(j + 1, K):
                if segment[j] > segment[k]:
                    contribution = factorial_K // 2
                else:
                    contribution = factorial_K // 2 - factorial(K - 1)
                total += contribution
        
        total -= original_inversions * factorial_K
    
    total_permutations = (N - K + 1) * factorial_K
    original_inversions = count_inversions(P)
    total += original_inversions * total_permutations
    
    return (total * mod_inverse(total_permutations)) % MOD

N, K = map(int, input().split())
P = list(map(int, input().split()))

result = calculate_expected_inversions(N, K, P)
print(result)