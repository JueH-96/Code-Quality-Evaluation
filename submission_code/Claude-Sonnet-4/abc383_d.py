import math

def sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, limit + 1) if is_prime[i]]

def solve(N):
    # We need primes up to N^(1/2) for p^2 * q^2 form
    # and up to N^(1/8) for p^8 form
    limit = int(math.sqrt(N)) + 1
    primes = sieve(limit)
    
    count = 0
    
    # Count numbers of form p^8
    for p in primes:
        if p**8 <= N:
            count += 1
        else:
            break
    
    # Count numbers of form p^2 * q^2 where p < q
    for i in range(len(primes)):
        p = primes[i]
        if p * p > math.sqrt(N):
            break
        
        for j in range(i + 1, len(primes)):
            q = primes[j]
            if p * p * q * q <= N:
                count += 1
            else:
                break
    
    return count

N = int(input())
print(solve(N))