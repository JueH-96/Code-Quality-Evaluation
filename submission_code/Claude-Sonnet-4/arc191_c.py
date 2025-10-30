def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def pow_mod(base, exp, mod):
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result

def multiplicative_order(a, m):
    if gcd(a, m) != 1:
        return -1
    
    order = 1
    current = a % m
    while current != 1:
        current = (current * a) % m
        order += 1
        if order > m:  # Safety check
            return -1
    return order

def solve(N):
    if N == 1:
        return 2, 1
    
    # Try to find a suitable M and A
    # Look for primes p where N divides (p-1)
    for p in range(2, min(1000, N * 100)):
        # Check if p is prime
        is_prime = True
        if p < 2:
            is_prime = False
        else:
            for i in range(2, int(p**0.5) + 1):
                if p % i == 0:
                    is_prime = False
                    break
        
        if not is_prime:
            continue
            
        if (p - 1) % N == 0:
            # Try to find a generator or element of order N
            for a in range(2, min(p, 100)):
                if gcd(a, p) == 1:
                    order = multiplicative_order(a, p)
                    if order == N:
                        return a, p
                    elif order > N and order % N == 0:
                        # a^(order/N) might have order N
                        new_a = pow_mod(a, order // N, p)
                        if multiplicative_order(new_a, p) == N:
                            return new_a, p
    
    # Fallback: try some other constructions
    # For small N, try M = N*k + 1 for small k
    for k in range(1, 100):
        M = N * k + 1
        for A in range(2, min(M, 100)):
            if gcd(A, M) == 1:
                order = multiplicative_order(A, M)
                if order == N:
                    return A, M
    
    # Another fallback
    return N + 1, N * 2 + 1

T = int(input())
for _ in range(T):
    N = int(input())
    A, M = solve(N)
    print(A, M)