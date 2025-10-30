from math import isqrt

def count_numbers_with_9_divisors(N):
    # For a prime p, p^8 has 9 divisors
    # For distinct primes p and q, p^2 * q^2 has 9 divisors
    
    # Generate primes up to N^(1/4)
    max_prime = isqrt(isqrt(N))
    
    # Sieve of Eratosthenes to generate primes
    is_prime = [True] * (max_prime + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, isqrt(max_prime) + 1):
        if is_prime[i]:
            for j in range(i*i, max_prime + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(max_prime + 1) if is_prime[i]]
    
    # Count numbers of the form p^8
    count_p8 = 0
    for p in primes:
        if p**8 <= N:
            count_p8 += 1
        else:
            break
    
    # Count numbers of the form p^2 * q^2 where p < q
    count_p2q2 = 0
    for i, p in enumerate(primes):
        p_squared = p**2
        limit_q = isqrt(N // p_squared)
        
        j = i + 1
        while j < len(primes) and primes[j] <= limit_q:
            count_p2q2 += 1
            j += 1
    
    return count_p8 + count_p2q2

def main():
    N = int(input())
    result = count_numbers_with_9_divisors(N)
    print(result)

if __name__ == "__main__":
    main()