import math
import bisect
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M = int(math.isqrt(N))
    
    primes = []
    if M >= 2:
        sieve_arr = [True] * (M + 1)
        sieve_arr[0] = False
        sieve_arr[1] = False
        sqrtM = int(math.isqrt(M))
        for i in range(2, sqrtM + 1):
            if sieve_arr[i]:
                for j in range(i * i, M + 1, i):
                    sieve_arr[j] = False
        primes = [i for i in range(2, M + 1) if sieve_arr[i]]
    
    count1 = 0
    for p in primes:
        power8 = p ** 8
        if power8 > N:
            break
        count1 += 1
        
    count2 = 0
    for p in primes:
        p2 = p * p
        if p2 > M:
            break
        q_max = M // p
        left_index = bisect.bisect_right(primes, p)
        right_index = bisect.bisect_right(primes, q_max)
        count_here = right_index - left_index
        if count_here > 0:
            count2 += count_here
            
    total = count1 + count2
    print(total)

if __name__ == '__main__':
    main()