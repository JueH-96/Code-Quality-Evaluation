from math import gcd
from collections import defaultdict
from functools import reduce

MOD = 998244353

def lcm(a, b):
    return a * b // gcd(a, b)

def get_factors(n):
    factors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            factors.append(i)
            if i * i != n:
                factors.append(n // i)
        i += 1
    return sorted(factors)

def get_coprime_pairs(n):
    factors = get_factors(n)
    pairs = []
    for p in factors:
        q = n // p
        if gcd(p, q) == 1:
            pairs.append((p, q))
    return pairs

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    # For each position, store possible values
    possible = [set() for _ in range(N)]
    
    # For each A[i], find all possible pairs (p,q) where p*q = A[i] and gcd(p,q) = 1
    pairs = []
    for i in range(N-1):
        pairs.append(get_coprime_pairs(A[i]))
    
    # For each pair in first position, try to build sequence
    def build_sequences(pos, curr_seq, curr_gcd):
        if pos == N:
            if curr_gcd == 1:  # Check if final gcd is 1
                return [curr_seq]
            return []
            
        if pos == 0:
            # First position: try all pairs for A[0]
            result = []
            for p, q in pairs[0]:
                result.extend(build_sequences(1, [p, q], gcd(p, q)))
            return result
            
        if pos == N-1:
            # Last position: only one value possible based on previous
            last = curr_seq[-1]
            new_gcd = gcd(curr_gcd, last)
            return build_sequences(pos+1, curr_seq, new_gcd)
            
        # Middle positions
        curr = curr_seq[-1]
        result = []
        for p, q in pairs[pos]:
            if q == curr:  # q should match previous number
                new_seq = curr_seq + [p]
                new_gcd = gcd(curr_gcd, p)
                result.extend(build_sequences(pos+1, new_seq, new_gcd))
        return result

    sequences = build_sequences(0, [], 0)
    
    # Calculate sum of scores
    total = 0
    for seq in sequences:
        score = 1
        for x in seq:
            score = (score * x) % MOD
        total = (total + score) % MOD
    
    print(total)

solve()