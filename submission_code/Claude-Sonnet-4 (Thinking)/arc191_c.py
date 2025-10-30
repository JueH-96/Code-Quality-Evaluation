def order(a, mod):
    """Compute the multiplicative order of a modulo mod."""
    from math import gcd
    if gcd(a, mod) != 1:
        return 0
    ord_candidate = 1
    current = a % mod
    while current != 1:
        current = (current * a) % mod
        ord_candidate += 1
        if ord_candidate > 10000:  # prevent infinite loop
            return 0
    return ord_candidate

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def solve(N):
    if N == 1:
        return 2, 1
    
    # Look for primes p such that N divides (p-1)
    for k in range(1, 1000):
        p = N * k + 1
        if p > 10**6:  # avoid large numbers to prevent timeouts
            break
        if is_prime(p):
            for a in range(2, min(100, p)):
                if order(a, p) == N:
                    return a, p
    
    # If no solution found, return a fallback
    # This should not happen given the problem constraints
    return 2, N + 1

T = int(input())
for _ in range(T):
    N = int(input())
    A, M = solve(N)
    print(A, M)