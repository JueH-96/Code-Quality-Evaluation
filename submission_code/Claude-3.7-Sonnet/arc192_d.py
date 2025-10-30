from math import gcd
from functools import reduce

def lcm(a, b):
    return (a * b) // gcd(a, b)

def gcd_list(numbers):
    return reduce(gcd, numbers)

def get_factorizations(A):
    pairs = []
    for P in range(1, int(A ** 0.5) + 1):
        if A % P == 0:
            Q = A // P
            if gcd(P, Q) == 1:
                pairs.append((P, Q))
                if P != Q:  # Avoid duplicates
                    pairs.append((Q, P))
    return pairs

def compute_sequence(factorizations, N):
    # Compute the sequence (u_i, v_i) such that S_i = S_1 * u_i / v_i
    u = [1]  # Numerator coefficients
    v = [1]  # Denominator coefficients
    
    for i in range(N - 1):
        P, Q = factorizations[i]
        new_u = u[-1] * Q
        new_v = v[-1] * P
        g = gcd(new_u, new_v)
        u.append(new_u // g)
        v.append(new_v // g)
    
    # Find the smallest valid S_1
    lcm_of_denominators = 1
    for denom in v:
        lcm_of_denominators = lcm(lcm_of_denominators, denom)
    
    S_1 = lcm_of_denominators
    
    # Compute the sequence
    S = [S_1]
    for i in range(1, N):
        S.append((S_1 * u[i]) // v[i])
    
    # Ensure gcd(S_1, S_2, ..., S_N) = 1
    g = gcd_list(S)
    if g > 1:
        S = [S_i // g for S_i in S]
    
    return S

def solve(N, A):
    mod = 998244353
    
    # Find all factorizations for each A_i
    factorizations_list = []
    for i in range(N - 1):
        factorizations_list.append(get_factorizations(A[i]))
    
    total_score = 0
    
    # Explore all possible combinations of factorizations
    def explore_factorizations(current_idx, current_factorizations):
        nonlocal total_score
        
        if current_idx == N - 1:
            # Compute the sequence
            S = compute_sequence(current_factorizations, N)
            
            # Compute the score
            score = 1
            for s in S:
                score = (score * s) % mod
            
            total_score = (total_score + score) % mod
            
            return
        
        for factorization in factorizations_list[current_idx]:
            current_factorizations.append(factorization)
            explore_factorizations(current_idx + 1, current_factorizations)
            current_factorizations.pop()
    
    explore_factorizations(0, [])
    
    return total_score

if __name__ == "__main__":
    # Input
    N = int(input())
    A = list(map(int, input().split()))
    
    # Solve
    result = solve(N, A)
    
    # Output
    print(result)