import sys

# MOD constant
MOD = 998244353

def mod_pow(base, exp, mod):
    res = 1
    base %= mod
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % mod
        base = (base * base) % mod
        exp //= 2
    return res

def mod_inverse(a, mod):
    # Use Fermat's Little Theorem as mod is prime
    return mod_pow(a, mod - 2, mod)

# Geometric sum S'(x, N) = sum_{k=0}^{N-1} x^k
def geometric_sum_0_to_N_minus_1(x, N, mod):
    if N == 0:
        return 0
    if x == 1:
        return N % mod
    
    # Sum = (x^N - 1) / (x - 1)
    x_N = mod_pow(x, N, mod)
    
    numerator = (x_N - 1 + mod) % mod
    denominator = (x - 1 + mod) % mod
    
    inv_denominator = mod_inverse(denominator, mod)
    
    result = (numerator * inv_denominator) % mod
    
    return result

def get_primes(m):
    primes = []
    if m < 2:
        return primes
    sieve = [True] * (m + 1)
    for p in range(2, m + 1):
        if sieve[p]:
            primes.append(p)
            # Optimized sieve: start marking from p*p
            for i in range(p * p, m + 1, p):
                sieve[i] = False
    return primes

def get_exponent(n, p):
    count = 0
    # Handle n=0 case? No, a >= 1 in the problem
    if p == 0 or p == 1: # Should not happen with valid primes
        return 0
    while n > 0 and n % p == 0:
        count += 1
        n //= p
    return count

# Calculate W[u] = sum_{a=1}^M prod_{i bit in u} v_i(a)
def calculate_W(M, primes):
    r = len(primes)
    max_mask = (1 << r)
    W = [0] * max_mask
    for a in range(1, M + 1):
        v_a = [get_exponent(a, p) for p in primes]
        for u in range(max_mask):
            prod = 1
            for i in range(r):
                if (u >> i) & 1:
                    prod *= v_a[i]
            W[u] += prod
    # No modulo needed yet for W before SOS as values are small
    return W

# Calculate W''[j] = sum_{a=1}^M prod_{u in j} (v_u(a)+2) mod MOD
def calculate_W_double_prime(M, primes):
    r = len(primes)
    max_mask = (1 << r)
    W_dp = [0] * max_mask
    for a in range(1, M + 1):
        v_a = [get_exponent(a, p) for p in primes]
        for u in range(max_mask):
            prod = 1
            for i in range(r):
                if (u >> i) & 1:
                    prod *= (v_a[i] + 2)
            W_dp[u] = (W_dp[u] + prod) % MOD
    return W_dp

# SOS transform
def sos_transform(arr, r):
    max_mask = (1 << r)
    transformed_arr = list(arr) # copy
    for i in range(r): # Iterate over bits
        for j in range(max_mask): # Iterate over masks
            if (j >> i) & 1:
                transformed_arr[j] = (transformed_arr[j] + transformed_arr[j ^ (1 << i)]) % MOD
    return transformed_arr

# Read input
N_large_str, M_str = sys.stdin.readline().split()
N_large = int(N_large_str)
M = int(M_str)

# Get primes up to M
primes = get_primes(M)
r = len(primes)
max_mask = (1 << r)

# Calculate W and its SOS transform hat_W
W = calculate_W(M, primes)
hat_W = sos_transform(W, r)

# Calculate W'' (W_dp)
W_dp = calculate_W_double_prime(M, primes) 

# Compute the total sum using the derived formula:
# Total sum = sum_{i=0}^{2^r-1} (-1)^{r-|i|} W''[i] S'(\hat{W}[i], N) mod MOD
total_sum = 0
for i in range(max_mask):
    # Calculate S'(\hat{W}[i], N)
    s_prime_val = geometric_sum_0_to_N_minus_1(hat_W[i], N_large, MOD)
    
    # Calculate term = W''[i] * S' mod MOD
    term = (W_dp[i] * s_prime_val) % MOD
    
    # Determine sign (-1)^(r - |i|)
    set_bits_count = bin(i).count('1')
    power_of_minus_1 = r - set_bits_count
    
    if power_of_minus_1 % 2 == 1: # if r - |i| is odd
        total_sum = (total_sum - term + MOD) % MOD
    else: # if r - |i| is even
        total_sum = (total_sum + term) % MOD

# Print the answer
print(total_sum)