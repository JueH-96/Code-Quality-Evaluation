# YOUR CODE HERE
import sys

# Increase recursion depth for potentially deep calculations if needed
# sys.setrecursionlimit(200000) 

MOD = 998244353

# Precompute factorials and their inverses up to N
# The maximum N given is 2*10^5
MAX_N_fact = 2 * 10**5 + 5 
fact = [1] * (MAX_N_fact + 1)
inv_fact = [1] * (MAX_N_fact + 1)

# Calculate factorials modulo MOD
for i in range(1, MAX_N_fact + 1):
    fact[i] = (fact[i-1] * i) % MOD

# Calculate modular inverse of the largest factorial using Fermat's Little Theorem
inv_fact[MAX_N_fact] = pow(fact[MAX_N_fact], MOD - 2, MOD)

# Calculate modular inverses of factorials down to 0!
for i in range(MAX_N_fact - 1, -1, -1):
    inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

def nCr_mod(n, r):
    """Computes nCr modulo MOD using precomputed factorials and inverses"""
    # Basic checks for validity of r
    if r < 0 or r > n:
        return 0
    # Base cases for combinations
    if r == 0 or r == n:
        return 1
    # Optimization: C(n, k) = C(n, n-k)
    # Choose the smaller value for r to potentially speed up calculation if factorials were not precomputed
    # But with precomputation, this step is not strictly needed for speed. It doesn't hurt.
    if r > n // 2:
        r = n - r
    
    # Calculate nCr = n! / (r! * (n-r)!) using modular arithmetic
    # nCr = n! * (r!)^{-1} * ((n-r)! )^{-1} (mod MOD)
    num = fact[n]
    den = (inv_fact[r] * inv_fact[n-r]) % MOD
    return (num * den) % MOD

def fast_pow(base, power):
    """Computes base^power % MOD efficiently using binary exponentiation"""
    result = 1
    while power > 0:
        if power % 2 == 1: # If power is odd
            result = (result * base) % MOD
        base = (base * base) % MOD # Square the base
        power //= 2 # Halve the power
    return result

def inverse(a):
    """Computes modular inverse of a modulo MOD using Fermat's Little Theorem"""
    # Assumes MOD is prime and a is not a multiple of MOD
    return fast_pow(a, MOD - 2)


def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    
    # A necessary condition for strong connectivity:
    # The first vertex (1) must be Black (S[0]=='B') AND
    # The last vertex (2N) must be White (S[2N-1]=='W').
    # If S[0]=='W', D(1)=0 because vertex 1 is White, cannot be paired as 'b' in (w,b) with b=1.
    # If S[2N-1]=='B', D(2N-1)=0 because vertex 2N is Black, cannot be paired as 'w' in (w,b) with w=2N.
    # The strong connectivity condition D(k)>=1 for all k=1..2N-1 fails if S[0]=='W' or S[2N-1]=='B'.
    if S[0] == 'W' or S[-1] == 'B':
        print(0)
        return

    # Calculate prefix balances p[i] = #W in S[0..i-1] - #B in S[0..i-1]
    p = [0] * (2 * N + 1)
    min_p = 0
    
    for i in range(2 * N):
        if S[i] == 'W':
            p[i+1] = p[i] + 1
        else: # S[i] == 'B'
            p[i+1] = p[i] - 1
        
        # Keep track of the minimum balance value
        min_p = min(min_p, p[i+1])

    # Sanity check: the final balance p[2N] must be 0 since there are N W's and N B's.
    if p[2*N] != 0:
         # This case should not happen based on problem statement guarantees.
         print(0) 
         return

    # Since S[0]=='B', p[1] = -1, which guarantees min_p <= -1.
    # The formula derived for the number of valid pairings is N! * (C(N, -min_p))^{-1} mod MOD
    # target_k is the required number of black vertices 'k' for the combination C(N, k).
    target_k = -min_p 
    
    # We know 1 <= target_k <= N.
    # Check if target_k is valid (it should be based on properties of balance sums).
    # Maximum value of B(i) - W(i) is N (when all B's appear before all W's). So target_k = max(B(i) - W(i)) <= N.
    # Since S[0]='B', p[1]=-1, min_p <= -1, so target_k >= 1.
    if not (1 <= target_k <= N):
         # Defensive coding: this case indicates an issue with the logic or input constraints.
         print(0) 
         return

    # Calculate C(N, target_k) mod MOD
    comb = nCr_mod(N, target_k)
    
    # Since 1 <= target_k <= N, C(N, target_k) >= N >= 1, so comb is guaranteed to be non-zero.
    if comb == 0:
         # Defensive coding: Should not happen for valid N, target_k.
         print(0)
         return

    # Calculate modular inverse of C(N, target_k)
    comb_inv = inverse(comb)
    
    # Final answer is N! * (C(N, target_k))^{-1} mod MOD
    # fact[N] contains N! mod MOD
    ans = (fact[N] * comb_inv) % MOD
    print(ans)
         
solve()