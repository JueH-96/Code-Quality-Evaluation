# YOUR CODE HERE
MOD = 998244353

def count_good_integers(N, M):
    def sum_of_divisors(n):
        total = 1
        p = 2
        while p * p <= n:
            if n % p == 0:
                count = 0
                while n % p == 0:
                    n //= p
                    count += 1
                total *= (p**(count + 1) - 1) // (p - 1)
            p += 1
        if n > 1:
            total *= (n**2 - 1) // (n - 1)
        return total

    def is_good(n):
        return sum_of_divisors(n) % 3 == 0

    good_count = 0
    for i in range(1, N + 1):
        if is_good(i):
            good_count += 1

    result = pow(good_count, M, MOD)
    return result

import sys
input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
print(count_good_integers(N, M))