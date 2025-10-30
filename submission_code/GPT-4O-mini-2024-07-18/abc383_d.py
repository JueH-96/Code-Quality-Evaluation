def count_numbers_with_nine_divisors(N):
    count = 0
    
    # A number has exactly 9 divisors if it can be expressed in one of the following forms:
    # 1. p^8 where p is a prime
    # 2. p^2 * q^2 where p and q are distinct primes
    
    # To find all primes up to a certain limit, we can use the Sieve of Eratosthenes
    import math
    
    # Limit for p^8
    limit_p8 = int(N**(1/8)) + 1
    # Limit for p^2 * q^2
    limit_p2q2 = int(N**(1/4)) + 1
    
    # Sieve of Eratosthenes to find all primes up to limit_p2q2
    sieve_limit = max(limit_p8, limit_p2q2)
    is_prime = [True] * (sieve_limit + 1)
    is_prime[0] = is_prime[1] = False
    
    for start in range(2, int(math.sqrt(sieve_limit)) + 1):
        if is_prime[start]:
            for multiple in range(start*start, sieve_limit + 1, start):
                is_prime[multiple] = False
    
    primes = [num for num, prime in enumerate(is_prime) if prime]
    
    # Count numbers of the form p^8
    for p in primes:
        p8 = p**8
        if p8 <= N:
            count += 1
        else:
            break
    
    # Count numbers of the form p^2 * q^2
    num_primes = len(primes)
    for i in range(num_primes):
        p2 = primes[i]**2
        if p2 > N:
            break
        for j in range(i + 1, num_primes):
            q2 = primes[j]**2
            if p2 * q2 <= N:
                count += 1
            else:
                break
    
    return count

import sys
input = sys.stdin.read

N = int(input().strip())
result = count_numbers_with_nine_divisors(N)
print(result)