import sys
import math

# Set a higher recursion limit as the DP depth can be significant
sys.setrecursionlimit(2 * 10**5) 

N, M = map(int, sys.stdin.readline().split())
MOD = 998244353

# --- Precomputation of Combinations (nCr_mod_p) ---
# C(e) = (e + M - 1) choose (M - 1)
# The maximum exponent 'e' for any prime p such that p^e <= N is floor(log_2(N)).
# For N = 10^10, log_2(10^10) approx 33.2. So max e is 33.
MAX_EXPONENT = 0
if N > 0: # N is guaranteed to be >= 1 by constraints
    # int(math.log2(N)) gives floor(log2(N)). Add 1 to be safe for cases like N=1 (log2(1)=0)
    # or N=2 (log2(2)=1). The largest exponent for p=2 should be floor(log2(N)).
    # N.bit_length() - 1 is floor(log2(N)) for N >= 1.
    MAX_EXPONENT = N.bit_length() - 1 
    if MAX_EXPONENT < 0: MAX_EXPONENT = 0 # Handle N=0, which is not allowed by constraints but good practice.

# Max value for n in nCr(n,r) is (MAX_EXPONENT + M - 1)
MAX_COMB_N = M + MAX_EXPONENT - 1

fact = [1] * (MAX_COMB_N + 1)
inv_fact = [1] * (MAX_COMB_N + 1)

for i in range(1, MAX_COMB_N + 1):
    fact[i] = (fact[i-1] * i) % MOD

# Precompute inverse factorials using Fermat's Little Theorem (a^(p-2) mod p)
# Since MOD is a prime number (998244353), this is valid.
inv_fact[MAX_COMB_N] = pow(fact[MAX_COMB_N], MOD - 2, MOD)
for i in range(MAX_COMB_N - 1, -1, -1):
    inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

def nCr_mod_p(n, r):
    if r < 0 or r > n:
        return 0
    # C(n,r) = n! / (r! * (n-r)!)
    numerator = fact[n]
    denominator = (inv_fact[r] * inv_fact[n-r]) % MOD
    return (numerator * denominator) % MOD

# Memoization for C(e) values to avoid recomputing for same e
C_e_memo = {}
def get_C_e(e):
    if e not in C_e_memo:
        C_e_memo[e] = nCr_mod_p(e + M - 1, M - 1)
    return C_e_memo[e]

# --- Sieve of Eratosthenes for primes up to sqrt(N) ---
SQRT_N = int(N**0.5)
# primes list will store primes up to SQRT_N
primes = []
if SQRT_N >= 2: # Only run sieve if SQRT_N is large enough to contain primes
    is_prime = [True] * (SQRT_N + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, SQRT_N + 1):
        if is_prime[p]:
            primes.append(p)
            for multiple in range(p*p, SQRT_N + 1, p):
                is_prime[multiple] = False

# --- Prepare distinct limit values for DP states ---
# The 'limit' argument in DP will always be of the form N // k.
# There are at most 2 * sqrt(N) such distinct values.
vals = set()
for k in range(1, SQRT_N + 1): # k iterates up to sqrt(N)
    vals.add(N // k)
    vals.add(k) # Also add k, as N//(N//k) = k. This covers values < sqrt(N).
vals = sorted(list(vals), reverse=True) # Sort in descending order. Larger values (like N) first.
# Map actual limit value to its index in the sorted `vals` list
val_to_idx = {v: i for i, v in enumerate(vals)}

# --- Main DP function ---
memo = {} # Memoization table for (idx, limit_idx) states

# solve(idx, limit_idx) computes the sum of f(P) for numbers P
# where P <= vals[limit_idx] and all prime factors of P are greater than or equal to primes[idx].
# It returns a tuple: (sum of f(P) for good P, sum of f(P) for bad P)
def solve(idx, limit_idx):
    current_limit = vals[limit_idx]

    # Base case: All small primes considered, or current prime is too large for current_limit.
    # This means any remaining number P must have prime factors > current_limit,
    # which implies P can only be 1 (since 1 has no prime factors).
    if idx >= len(primes) or primes[idx] > current_limit:
        # P = 1 contributes f(1) = C(0) = 1.
        # sigma(1) = 1, which is not divisible by 3. So 1 is a "bad" number.
        if current_limit >= 1:
            return (0, get_C_e(0)) # (good_sum, bad_sum)
        else: # current_limit is 0, no numbers possible
            return (0, 0)

    key = (idx, limit_idx)
    if key in memo:
        return memo[key]

    total_good_sum = 0
    total_bad_sum = 0

    p = primes[idx]

    # Term 1: p is not a factor of P (its exponent e = 0)
    # This case corresponds to P = 1 * Q, where Q has prime factors >= primes[idx+1].
    # sigma(p^0) = sigma(1) = 1, which is a "bad part" (not divisible by 3).
    # So, if P = 1 * Q, then P is good iff Q is good. P is bad iff Q is bad.
    # C(0) = 1, so the factors `get_C_e(0)` are effectively 1.
    next_good_sum_Q, next_bad_sum_Q = solve(idx + 1, limit_idx) # Q <= current_limit
    total_good_sum = (total_good_sum + get_C_e(0) * next_good_sum_Q) % MOD
    total_bad_sum = (total_bad_sum + get_C_e(0) * next_bad_sum_Q) % MOD

    # Term 2: p is a factor of P with exponent e >= 1
    power_p = p
    exponent = 1
    while power_p <= current_limit:
        val_C_e = get_C_e(exponent)

        # Determine if sigma(p^e) is divisible by 3 (i.e., p^e is a "good part")
        is_p_e_good_part = False
        if p % 3 == 1:
            # sigma(p^e) = sum_{i=0 to e} p^i. If p=1(mod 3), then sigma(p^e) = (e+1)(mod 3).
            # It's a "good part" if (e+1) is divisible by 3.
            if (exponent + 1) % 3 == 0:
                is_p_e_good_part = True
        elif p % 3 == 2:
            # sigma(p^e) = sum_{i=0 to e} p^i. If p=2(mod 3), then p=-1(mod 3).
            # sigma(p^e) = sum_{i=0 to e} (-1)^i (mod 3). This sum is 0 (mod 3) iff e is odd.
            # It's a "good part" if e is odd.
            if exponent % 2 == 1:
                is_p_e_good_part = True
        # If p=3, sigma(3^e) = 1+3+...+3^e = 1 (mod 3) for e >= 0.
        # So p^e is never a "good part" if p=3. `is_p_e_good_part` correctly remains False.

        # The remaining part Q must satisfy Q <= current_limit / p^e,
        # and its prime factors must be >= primes[idx+1].
        next_limit_val_for_Q = current_limit // power_p
        next_limit_idx_for_Q = val_to_idx[next_limit_val_for_Q]
        
        next_good_sum_Q, next_bad_sum_Q = solve(idx + 1, next_limit_idx_for_Q)

        if is_p_e_good_part:
            # If p^e contributes a 0 (mod 3) factor to sigma(P), then P=p^e*Q is good
            # regardless of whether sigma(Q) is divisible by 3.
            total_good_sum = (total_good_sum + val_C_e * (next_good_sum_Q + next_bad_sum_Q)) % MOD
        else: # p^e is a "bad part" (sigma(p^e) is not 0 mod 3)
            # If p^e is a "bad part", then P=p^e*Q is good only if Q is good.
            # P=p^e*Q is bad only if Q is bad.
            total_good_sum = (total_good_sum + val_C_e * next_good_sum_Q) % MOD
            total_bad_sum = (total_bad_sum + val_C_e * next_bad_sum_Q) % MOD

        # Check for potential overflow before multiplying power_p by p
        # `power_p * p <= current_limit` is equivalent to `power_p <= current_limit / p`
        if power_p > current_limit // p: 
            break
        power_p *= p
        exponent += 1

    memo[key] = (total_good_sum, total_bad_sum)
    return (total_good_sum, total_bad_sum)

# The final answer is the sum of f(P) for good P where P <= N.
# This corresponds to the first element of the tuple returned by solve(0, val_to_idx[N]).
final_good_sum, _ = solve(0, val_to_idx[N])
print(final_good_sum)