from math import gcd
from functools import reduce

MOD = 998244353

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def solve(N, A):
    # Calculate the total number of outcomes
    total_outcomes = reduce(lambda x, y: x * y % MOD, A, 1)

    # Calculate the number of favorable outcomes
    favorable_outcomes = 0
    for i in range(1 << N):
        subset_sum = 0
        subset_product = 1
        for j in range(N):
            if i & (1 << j):
                subset_sum += 1
                subset_product = subset_product * A[j] % MOD
        if subset_sum == 10:
            favorable_outcomes = (favorable_outcomes + subset_product) % MOD
        elif subset_sum > 10:
            break

    # Calculate the probability
    probability = favorable_outcomes * mod_inverse(total_outcomes) % MOD

    return probability

N = int(input())
A = list(map(int, input().split()))

print(solve(N, A))