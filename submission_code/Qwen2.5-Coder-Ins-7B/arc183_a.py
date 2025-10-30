# YOUR CODE HERE
import sys
from math import factorial

def factorial_mod(n, mod):
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result = (result * i) % mod
    return result

def mod_inverse(a, mod):
    return pow(a, mod - 2, mod)

def comb_mod(n, k, mod):
    if k > n:
        return 0
    return (factorial_mod(n, mod) * mod_inverse(factorial_mod(k, mod), mod) * mod_inverse(factorial_mod(n - k, mod), mod)) % mod

def nth_permutation(n, k, mod):
    result = []
    remaining = list(range(1, n + 1))
    for i in range(n, 0, -1):
        index = (k - 1) // comb_mod(i - 1, 1, mod)
        result.append(remaining.pop(index))
        k -= index * comb_mod(i - 1, 1, mod)
    return result

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = factorial(N) // (factorial(K) ** N)
    target = (S + 1) // 2
    mod = 10**9 + 7
    result = nth_permutation(N, target, mod)
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()