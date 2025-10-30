import sys
import math
from math import factorial
from collections import defaultdict

def main():
    import sys
    sys.setrecursionlimit(1000000)
    N, K = map(int, sys.stdin.read().split())
    counts = [K] * N
    total = N * K
    # Compute the total number of sequences S = (NK)! / (K!)^N
    # To compute floor((S+1)/2), we compute S and then integer divide
    # However, (NK)! can be very large, use logarithm to compute S
    # But we need exact S to compute floor((S+1)/2)
    # Instead, use multiplicative formula

    # To compute S, we use log factorial to compute log(S)
    # But we need exact S, so use multiplicative formula with caching
    # Precompute factorials up to NK
    # However, for large N and K, it's too slow, use math.factorial

    # Compute S
    # To prevent recomputing factorials multiple times, cache them
    fact_cache = {}
    def fact(n):
        if n in fact_cache:
            return fact_cache[n]
        fact_cache[n] = math.factorial(n)
        return fact_cache[n]

    # Compute (NK)!
    S = fact(total)
    # Compute (K!)^N
    denom = fact(K) ** N
    S = S // denom
    target = (S + 1) // 2

    sequence = []
    for _ in range(total):
        for num in range(1, N+1):
            if counts[num-1] == 0:
                continue
            counts[num-1] -=1
            # Compute number of sequences with current counts
            rem = total - len(sequence) -1
            numerator = fact(rem)
            denominator = 1
            for c in counts:
                denominator *= fact(c)
            count = numerator // denominator
            if count >= target:
                sequence.append(num)
                break
            else:
                target -= count
                counts[num-1] +=1
    print(' '.join(map(str, sequence)))

if __name__ == "__main__":
    main()