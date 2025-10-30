def sum_of_divisors(n):
    total = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            total += i
            if i != n // i:
                total += n // i
    return total

def is_good_integer(n):
    return sum_of_divisors(n) % 3 == 0

def count_good_integers(N):
    count = 0
    for i in range(1, N + 1):
        if is_good_integer(i):
            count += 1
    return count

def mod_exp(base, exp, mod):
    result = 1
    base = base % mod
    while exp > 0:
        if (exp % 2) == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

def count_sequences(N, M):
    good_count = count_good_integers(N)
    mod = 998244353
    return (mod_exp(good_count, M, mod)) % mod

import sys
input = sys.stdin.read
N, M = map(int, input().strip().split())
print(count_sequences(N, M))