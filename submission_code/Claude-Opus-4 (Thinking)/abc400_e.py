def sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

# Precompute primes up to 10^6
primes = sieve(10**6)

Q = int(input())
for _ in range(Q):
    A = int(input())
    max_400 = -1
    
    for i in range(len(primes)):
        p1 = primes[i]
        p1_sq = p1 * p1
        if p1_sq > A:
            break
        
        for j in range(len(primes)):
            if i == j:
                continue
            p2 = primes[j]
            p2_sq = p2 * p2
            if p1_sq * p2_sq > A:
                continue
            
            # Find the largest p1^(2a) * p2^(2b) <= A
            p1_pow = p1_sq
            while p1_pow <= A:
                p2_pow = p2_sq
                while p1_pow * p2_pow <= A:
                    max_400 = max(max_400, p1_pow * p2_pow)
                    p2_pow *= p2_sq
                p1_pow *= p1_sq
    
    print(max_400)