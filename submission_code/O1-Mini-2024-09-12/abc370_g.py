# YOUR CODE HERE
import sys, math
from math import log
import sys
sys.setrecursionlimit(1000000)

MOD = 998244353

def main():
    import sys
    import math
    from math import isqrt

    N, M = map(int, sys.stdin.read().split())

    # Sieve of Eratosthenes up to 100
    def sieve(n):
        sieve = [True]*(n+1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        primes = [i for i, is_p in enumerate(sieve) if is_p]
        return primes

    primes = sieve(100)  # Adjust sieve limit as needed
    # Precompute factorial and inverse factorial
    max_fact = M + 100
    fact = [1]*(max_fact +1)
    for i in range(1, max_fact +1):
        fact[i] = fact[i-1]*i % MOD
    inv_fact = [1]*(max_fact +1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
    for i in range(max_fact, 0, -1):
        inv_fact[i-1] = inv_fact[i]*i % MOD
    def comb(a, b):
        if a < b or b <0:
            return 0
        return fact[a]*inv_fact[b]%MOD * inv_fact[a-b]%MOD

    answer = 0

    def backtrack(index, current_product, has_A, has_B, multiplier):
        nonlocal answer
        if current_product > N:
            return
        if index == len(primes):
            if has_A or has_B:
                answer = (answer + multiplier) % MOD
            return
        p = primes[index]
        max_a = 0
        temp = current_product
        while temp * p <= N:
            temp *= p
            max_a +=1
        for a in range(0, max_a+1):
            new_product = current_product * (p**a)
            if new_product > N:
                break
            new_has_A = has_A
            new_has_B = has_B
            if p %3 ==1 and a %3 ==2:
                new_has_A = True
            if p %3 ==2 and a %2 ==1:
                new_has_B = True
            new_multiplier = multiplier
            if a >0:
                new_multiplier = new_multiplier * comb(a + M -1, a) % MOD
            backtrack(index+1, new_product, new_has_A, new_has_B, new_multiplier)

    backtrack(0, 1, False, False, 1)

    print(answer)

if __name__ == "__main__":
    main()