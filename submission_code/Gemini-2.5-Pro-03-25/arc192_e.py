# YOUR CODE HERE
import sys

# Increase recursion depth if needed, although this solution is iterative
# sys.setrecursionlimit(3 * 10**6 + 10) 

MOD = 998244353

# Precompute factorials and inverse factorials for combinations modulo MOD
# Maximum value for n in C(n, k) is W+H+3 based on the formula G(x, y) uses.
MAX_N = 2 * 10**6 + 10 # W+H+3 can be up to 2*10^6 + 3, add some buffer
fact = [1] * (MAX_N + 1)
inv_fact = [1] * (MAX_N + 1)

# Calculate factorials modulo MOD
for i in range(1, MAX_N + 1):
    fact[i] = (fact[i-1] * i) % MOD

# Calculate inverse factorial of MAX_N using Fermat's Little Theorem (since MOD is prime)
# a^(MOD-2) % MOD is the modular inverse of a % MOD
inv_fact[MAX_N] = pow(fact[MAX_N], MOD - 2, MOD)

# Calculate inverse factorials iteratively from MAX_N down to 0
# inv_fact[i] = inv_fact[i+1] * (i+1) % MOD
for i in range(MAX_N - 1, -1, -1):
    inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD

# Function to compute nCr mod P using precomputed values
def nCr_mod(n, r):
    """Computes nCr modulo MOD efficiently."""
    if r < 0 or r > n:
        return 0
    # Base cases
    if r == 0 or r == n:
        return 1
    # Optimization: C(n, k) = C(n, n-k). Compute with smaller k.
    if r > n // 2:
        r = n - r
    
    # Calculate nCr = n! / (r! * (n-r)!) using modular arithmetic
    # nCr = fact[n] * inv_fact[r] * inv_fact[n-r] % MOD
    num = fact[n]
    den = (inv_fact[r] * inv_fact[n-r]) % MOD
    return (num * den) % MOD

# Define F(x, y) = C(x+y+1, x+1). 
# F(x, y) counts the number of paths consisting of right/up moves from (0,0) to (x,y).
def F(x, y):
     """Counts paths from (0,0) to (x,y)."""
     if x < 0 or y < 0:
         return 0
     # C(x+y+1, x+1) mod MOD
     # Note that C(x+y+1, x+1) = C(x+y+1, y)
     val = nCr_mod(x + y + 1, x + 1)
     return val

# Read input values
W, H, L, R, D, U = map(int, sys.stdin.readline().split())

# The problem asks for the total number of valid paths. A path is defined by its sequence of visited blocks.
# According to interpretations and similar problems (like AtCoder ABC154 F), 
# the total number of paths can be calculated as the sum over all allowed blocks (x, y) 
# of the number of paths from (0,0) to (x,y).
# This sum over a region can be calculated efficiently using a 2D prefix sum structure on F(x,y).
# Define G(x, y) = Sum_{i=0..x, j=0..y} F(i, j).
# G(x, y) = Sum_{i=0..x} Sum_{j=0..y} C(i+j+1, i+1)
# Applying hockey-stick identity twice, we find: G(x, y) = C(x+y+3, x+2).
def G(x, y):
    """Computes the 2D prefix sum of F(i, j) up to (x, y)."""
    if x < 0 or y < 0:
        return 0
    # Returns C(x+y+3, x+2) mod MOD
    # Note that C(x+y+3, x+2) = C(x+y+3, y+1)
    return nCr_mod(x + y + 3, x + 2)

# The total required sum is the sum of F(x, y) over the allowed region S.
# S is the set of blocks, which is the grid [0, W] x [0, H] minus the hole region [L, R] x [D, U].
# We can compute the sum over S using the principle of inclusion-exclusion for 2D sums.
# Sum over S = Sum over total grid - Sum over Hole region

# Sum F(x, y) over the total grid [0, W] x [0, H] is G(W, H).
total_sum_F_full_grid = G(W, H)

# Sum F(x, y) over the hole region [L, R] x [D, U].
# This sum is calculated using G functions:
# Sum over [L, R] x [D, U] = G(R, U) - G(L - 1, U) - G(R, D - 1) + G(L - 1, D - 1).
# Note the coordinates are inclusive: [L, R] means x from L to R. The range starts at index L.
# The prefix sum G(x, y) sums up to index x and y.
# For sum over range [x1, x2], we need G(x2) - G(x1-1).
# So for range [L, R], the x coordinates are covered by G(R) - G(L-1).
# Similarly for y coordinates D to U, covered by G(U) - G(D-1).
hole_sum_F = (G(R, U) - G(L - 1, U) - G(R, D - 1) + G(L - 1, D - 1)) % MOD

# The final result is Sum over S = Sum over total grid - Sum over Hole region.
# Need to handle potential negative results from subtraction by adding MOD.
result = (total_sum_F_full_grid - hole_sum_F + MOD) % MOD

# Print the final answer modulo MOD
print(result)