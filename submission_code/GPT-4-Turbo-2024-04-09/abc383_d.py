import sys
import math

def count_divisors(n):
    count = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            if i * i == n:
                count += 1
            else:
                count += 2
    return count

def main():
    input = sys.stdin.read
    N = int(input().strip())
    
    count = 0
    # To have exactly 9 divisors, the number must be of the form:
    # p^8 or p^2 * q^2 or p^4 * q where p and q are distinct primes
    # p^8 <= N => p <= N^(1/8)
    # p^2 * q^2 <= N => p, q <= N^(1/4)
    # p^4 * q <= N => p <= N^(1/4), q <= N^(1/2)
    
    # Upper bounds for p and q calculations
    max_p_8 = int(N ** (1/8))
    max_p_4 = int(N ** (1/4))
    max_p_2 = int(N ** (1/2))
    
    # To efficiently check for primes, use a sieve
    sieve_size = max_p_2 + 1
    is_prime = [True] * sieve_size
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(math.sqrt(sieve_size)) + 1):
        if is_prime[i]:
            for j in range(i * i, sieve_size, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Count numbers of the form p^8
    for p in primes:
        if p > max_p_8:
            break
        if p ** 8 <= N:
            count += 1
    
    # Count numbers of the form p^2 * q^2
    for i in range(len(primes)):
        p = primes[i]
        if p ** 2 > max_p_4:
            break
        for j in range(i + 1, len(primes)):
            q = primes[j]
            if q ** 2 > max_p_4:
                break
            if p ** 2 * q ** 2 <= N:
                count += 1
    
    # Count numbers of the form p^4 * q
    for i in range(len(primes)):
        p = primes[i]
        if p ** 4 > max_p_4:
            break
        for q in primes:
            if q > max_p_2:
                break
            if p == q:
                continue
            if p ** 4 * q <= N:
                count += 1
    
    print(count)

if __name__ == "__main__":
    main()