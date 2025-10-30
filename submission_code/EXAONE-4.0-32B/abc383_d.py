import math
import bisect

def main():
    N = int(input().strip())
    T = math.isqrt(N)
    
    if T < 2:
        primes = []
    else:
        sieve = [True] * (T + 1)
        sieve[0] = False
        sieve[1] = False
        sqrt_T = math.isqrt(T)
        for i in range(2, sqrt_T + 1):
            if sieve[i]:
                for j in range(i * i, T + 1, i):
                    sieve[j] = False
        primes = [i for i in range(2, T + 1) if sieve[i]]
    
    count1 = 0
    for p in primes:
        p8 = p ** 8
        if p8 <= N:
            count1 += 1
        else:
            break
    
    count2 = 0
    n = len(primes)
    for i in range(n):
        p = primes[i]
        if p * p > T:
            break
        q_max = T // p
        j = bisect.bisect_right(primes, q_max)
        count_here = j - (i + 1)
        if count_here > 0:
            count2 += count_here
    
    print(count1 + count2)

if __name__ == "__main__":
    main()