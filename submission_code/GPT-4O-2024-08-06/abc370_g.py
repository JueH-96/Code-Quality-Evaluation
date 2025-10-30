# YOUR CODE HERE
def sum_of_divisors(x):
    total = 0
    for i in range(1, int(x**0.5) + 1):
        if x % i == 0:
            total += i
            if i != x // i:
                total += x // i
    return total

def is_good_integer(x):
    return sum_of_divisors(x) % 3 == 0

def count_good_sequences(N, M, MOD=998244353):
    good_integers = []
    for i in range(1, N + 1):
        if is_good_integer(i):
            good_integers.append(i)
    
    # The number of sequences of length M with product g is g^(M-1)
    result = 0
    for g in good_integers:
        result = (result + pow(g, M - 1, MOD)) % MOD
    
    return result

import sys
input = sys.stdin.read
data = input().strip()
N, M = map(int, data.split())

print(count_good_sequences(N, M))