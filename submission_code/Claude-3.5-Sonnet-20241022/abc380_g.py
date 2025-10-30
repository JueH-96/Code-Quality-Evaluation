from fractions import Fraction
import itertools

def count_inversions(perm):
    inv = 0
    n = len(perm)
    for i in range(n):
        for j in range(i+1, n):
            if perm[i] > perm[j]:
                inv += 1
    return inv

def mod_inverse(a, m):
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    _, x, _ = extended_gcd(a, m)
    return (x % m + m) % m

def solve():
    MOD = 998244353
    
    # Read input
    N, K = map(int, input().split())
    P = list(map(int, input().split()))
    
    # If K=1 or N=1, no change possible
    if K == 1 or N == 1:
        return count_inversions(P)
    
    # Calculate total number of possible positions and permutations
    pos_count = N - K + 1
    perm_count = 1
    for i in range(K):
        perm_count *= (i + 1)
    
    # Calculate expected value
    total = Fraction(0)
    
    # For each possible starting position
    for start_pos in range(N - K + 1):
        # For each possible permutation of the K elements
        segment = P[start_pos:start_pos + K]
        for perm in itertools.permutations(segment):
            # Create new permutation
            new_P = P[:start_pos] + list(perm) + P[start_pos + K:]
            # Count inversions
            inv_count = count_inversions(new_P)
            # Add to total with appropriate probability
            total += Fraction(inv_count, pos_count * perm_count)
    
    # Convert fraction to modulo
    num = total.numerator
    den = total.denominator
    
    # Calculate (num * den^(-1)) % MOD
    den_inv = mod_inverse(den, MOD)
    result = (num * den_inv) % MOD
    
    return result

print(solve())