import bisect
import math

def sieve(n):
    if n < 2:
        return []
    sieve_list = [True] * (n + 1)
    sieve_list[0] = sieve_list[1] = False
    for i in range(2, int(math.sqrt(n)) + 1):
        if sieve_list[i]:
            sieve_list[i*i : n+1 : i] = [False] * len(sieve_list[i*i : n+1 : i])
    primes = [i for i, is_prime in enumerate(sieve_list) if is_prime]
    return primes

def main():
    N = int(input())
    X = math.isqrt(N)
    primes = sieve(X)
    
    count1 = 0
    for p in primes:
        if p ** 8 <= N:
            count1 += 1
        else:
            break
    
    count2 = 0
    for i in range(len(primes)):
        p = primes[i]
        max_q = X // p
        if max_q <= p:
            continue
        idx = bisect.bisect_right(primes, max_q)
        cnt = idx - (i + 1)
        if cnt > 0:
            count2 += cnt
    
    print(count1 + count2)

if __name__ == "__main__":
    main()