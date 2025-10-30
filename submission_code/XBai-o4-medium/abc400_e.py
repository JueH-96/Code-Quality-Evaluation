import math
import sys
import bisect

def main():
    input = sys.stdin.read().split()
    Q = int(input[0])
    queries = list(map(int, input[1:Q+1]))
    
    max_k = 10**6
    spf = list(range(max_k + 1))
    for i in range(2, int(math.isqrt(max_k)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_k + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    valid_ks = []
    for n in range(2, max_k + 1):
        num = n
        primes = set()
        while num != 1:
            p = spf[num]
            primes.add(p)
            while num % p == 0:
                num //= p
        if len(primes) == 2:
            valid_ks.append(n)
    
    for A in queries:
        S = math.isqrt(A)
        pos = bisect.bisect_right(valid_ks, S)
        K = valid_ks[pos - 1]
        print(K * K)

if __name__ == "__main__":
    main()