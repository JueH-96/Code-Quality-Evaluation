import sys
import math
import bisect

def sieve(n):
    sieve = bytearray([True]) * (n+1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(n)) +1):
        if sieve[i]:
            sieve[i*i:n+1:i] = b'\x00' * len(sieve[i*i:n+1:i])
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    return primes

def main():
    import sys
    N_str = sys.stdin.read().strip()
    N = int(N_str)
    
    if N < 1:
        print(0)
        return
    
    sqrtN = int(math.isqrt(N))
    primes = sieve(sqrtN)
    
    # Count1: primes p where p^8 <= N
    max_p8 = int(N ** (1/8))
    count1 = 0
    for p in primes:
        if p > max_p8:
            break
        if p**8 <= N:
            count1 +=1
    # Count2: number of pairs p < q with p*q <= sqrtN
    count2 = 0
    num_primes = len(primes)
    for i in range(num_primes):
        p = primes[i]
        if p > sqrtN:
            break
        limit = sqrtN // p
        # Find the index of the largest prime <= limit
        j = bisect.bisect_right(primes, limit) -1
        if j > i:
            count2 += (j - i)
    total = count1 + count2
    print(total)

if __name__ == "__main__":
    main()