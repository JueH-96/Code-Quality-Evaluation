from math import gcd
from functools import reduce

def factorize(n):
    # Find all ways to factorize n as p * q where gcd(p, q) = 1
    factors = []
    for p in range(1, n + 1):
        if n % p == 0:
            q = n // p
            if gcd(p, q) == 1:
                factors.append((p, q))
    return factors

def lcm(a, b):
    return a * b // gcd(a, b)

def compute_good_sequence(P, Q, N):
    # Compute the sequence S in terms of S_1
    # S_i = S_1 * prod(Q_j for j < i) / prod(P_j for j < i)
    
    # Compute the fractions S_i / S_1
    fractions = [(1, 1)]  # (numerator, denominator)
    
    for i in range(1, N):
        num = fractions[-1][0] * Q[i-1]
        den = fractions[-1][1] * P[i-1]
        g = gcd(num, den)
        fractions.append((num // g, den // g))
    
    # S_1 must be divisible by lcm of all denominators
    L = reduce(lcm, [frac[1] for frac in fractions])
    
    # Compute the sequence S
    S = []
    for num, den in fractions:
        S.append(L * num // den)
    
    # Check if gcd(S) = 1
    g = reduce(gcd, S)
    if g == 1:
        return S
    else:
        return None

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    # Generate all possible factorizations
    factorizations = []
    for a in A:
        factorizations.append(factorize(a))
    
    # Iterate over all combinations
    total_score = 0
    MOD = 998244353
    
    def dfs(index, P, Q):
        nonlocal total_score
        
        if index == N - 1:
            # We have a complete choice of (P_i, Q_i)
            S = compute_good_sequence(P, Q, N)
            if S is not None:
                score = 1
                for s in S:
                    score = (score * (s % MOD)) % MOD
                total_score = (total_score + score) % MOD
            return
        
        for p, q in factorizations[index]:
            dfs(index + 1, P + [p], Q + [q])
    
    dfs(0, [], [])
    
    print(total_score)

solve()