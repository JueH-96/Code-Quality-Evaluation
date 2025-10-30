import math
import bisect

def sieve(n):
    if n < 2:
        return []
    sieve_list = [True] * (n + 1)
    sieve_list[0] = sieve_list[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve_list[i]:
            sieve_list[i*i::i] = [False] * len(sieve_list[i*i::i])
    primes = [i for i, is_prime in enumerate(sieve_list) if is_prime]
    return primes

def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    K = math.isqrt(N)
    primes = sieve(K)
    
    # Calculate count_A (numbers p^8 <= N)
    count_A = 0
    for p in primes:
        if p ** 8 <= N:
            count_A += 1
    
    # Calculate count_B (pairs p*q <= K, p < q)
    count_B = 0
    for i in range(len(primes)):
        p = primes[i]
        if p > K // p:
            break
        max_q = K // p
        pos = bisect.bisect_right(primes, max_q)
        count = pos - (i + 1)
        if count > 0:
            count_B += count
    
    print(count_A + count_B)

if __name__ == "__main__":
    main()