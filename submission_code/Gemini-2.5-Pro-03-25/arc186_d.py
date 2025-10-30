# YOUR CODE HERE
import sys

# Set higher recursion depth limit for potentially deep recursion
# The default is often 1000, which might be too small for N=3e5
# Try setting a large limit, catch error if OS prevents it
try:
    # Set recursion depth limit slightly larger than N, plus some buffer
    sys.setrecursionlimit(300000 + 5000) 
except OverflowError: # Handle potential OS limits on stack size
     # Can't print to stderr in contest, just ignore or proceed cautiously
     pass 


MOD = 998244353

N = 0
A = []
P = [] # P[k] = number of Polish sequences of length k
# CheckCache stores results of check_polish_prefix for sequence A
CheckCache = {} 
# memo_solve and memo_concat store results for the main DP functions
memo_solve = {}
memo_concat = {}

# Factorials and inverse factorials for combinations
fact = []
inv_fact = []

# Precompute factorials and their modular inverses up to max_val
def precompute_combinatorics(max_val, mod):
    global fact, inv_fact
    # Ensure arrays handle non-negative max_val
    if max_val < 0: max_val = 0
    fact = [1] * (max_val + 1)
    inv_fact = [1] * (max_val + 1)
    for i in range(1, max_val + 1):
        fact[i] = (fact[i - 1] * i) % mod
    
    if max_val >= 0: # Avoid index error if max_val < 0
        # Compute modular inverse using Fermat's Little Theorem (mod must be prime)
        inv_fact[max_val] = pow(fact[max_val], mod - 2, mod)
        # Compute other inverse factorials iteratively
        for i in range(max_val - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod

# Compute nCr modulo mod using precomputed factorials and inverses
def nCr_mod(n, r, mod):
    if r < 0 or r > n:
        return 0
    if r == 0 or r == n:
        return 1
    # Optimization: Use symmetry C(n, r) = C(n, n-r)
    if r > n // 2:
        r = n - r
    
    # Check if factorials arrays are large enough. This indicates an error if check fails.
    if n >= len(fact):
         # Error state, precomputation bounds were too small.
         # In a contest, might return 0 or raise error. Depends on expected behavior.
         return 0 

    # Calculate nCr using formula: n! / (r! * (n-r)!)
    num = fact[n]
    den = (inv_fact[r] * inv_fact[n - r]) % mod
    return (num * den) % mod

# Precompute P[k], the number of Polish sequences of length k
def precompute_P(max_k, mod):
    global P
    P = [0] * (max_k + 1)
    if max_k == 0: return
    
    # Need factorials up to 2*max_k for combinations C(2k-2, k-1)
    precompute_combinatorics(2 * max_k, mod) 

    # Precompute modular inverses of k needed for P[k] formula
    inv_k = [0] * (max_k + 1)
    if max_k >= 1:
      inv_k[1] = 1 # Inverse of 1 is 1
      for k in range(2, max_k + 1):
        inv_k[k] = pow(k, mod - 2, mod) # Modular inverse using Fermat's Little Theorem

    P[0] = 0 # Length 0 sequence not possible/defined as Polish
    if max_k >= 1:
       P[1] = 1 # Base case: sequence (0) is Polish, length 1

    # Compute P[k] for k >= 2 using Catalan number relation
    for k in range(2, max_k + 1):
        # P[k] = C_{k-1} = (1/k) * C(2k-2, k-1)
        # Check arguments for nCr validity
        if 2*k - 2 >= k - 1:
           comb = nCr_mod(2*k - 2, k - 1, mod)
           P[k] = (inv_k[k] * comb) % mod
        else: # Should not happen for k >= 2
           P[k] = 0


# Function check_polish_prefix(start_idx): Checks if A[start_idx:] starts with a Polish sequence.
# Returns tuple (is_polish: bool, length_consumed: int). Uses memoization via CheckCache.
def check_polish_prefix(start_idx):
    state = start_idx
    if state in CheckCache:
        return CheckCache[state]

    # Base case: index out of bounds
    if start_idx >= N:
        return False, 0 

    # Get the first element V1 = A[start_idx]
    V1 = A[start_idx]
    
    # Base case: If V1 is 0, sequence must be (0), length 1.
    if V1 == 0:
        CheckCache[state] = (True, 1)
        return True, 1

    # Recursive case: V1 > 0. Needs V1 Polish subsequences.
    current_idx = start_idx + 1
    total_len = 1 # Start with length 1 for V1 itself
    
    possible = True
    # Iterate V1 times to find V1 Polish subsequences
    for _ in range(V1):
        # Check if there are enough elements left in A
        if current_idx >= N:
            possible = False # Not enough elements
            break 
        
        # Recursively check the subsequence starting at current_idx
        is_sub_polish, sub_len = check_polish_prefix(current_idx)
        
        # If subproblem fails or returns invalid length
        if not is_sub_polish or sub_len == 0: 
            possible = False
            break
        
        # Advance index and update total length
        current_idx += sub_len
        total_len += sub_len
        # Safety check (shouldn't be needed if sub_len > 0 always for Polish)
        if sub_len == 0: 
             possible = False; break 

    # Memoize the result (even failures)
    if not possible:
        CheckCache[state] = (False, total_len) # Store partial length on failure
        return False, total_len
    
    # If successful, store True and the total length consumed
    CheckCache[state] = (True, total_len)
    return True, total_len

# Precompute check_polish_prefix values for all indices of A using DP approach.
# This relies on the property that total work is O(N).
def precompute_check():
    for i in range(N - 1, -1, -1):
        check_polish_prefix(i)


# Precompute SumCountPartitions(v, k) = sum P(k1)*...*P(kv) where sum ki = k.
# Uses O(N^2) Dynamic Programming based on recurrence: c_{v,k} = c_{v-1, k} - c_{v-2, k-1}
# Note: This O(N^2) step is the likely performance bottleneck for large N.
C_table = []

def precompute_SumCountPartitions(max_v, max_k, mod):
    global C_table
    # Initialize table with dimensions (max_v+1) x (max_k+1)
    C_table = [[0] * (max_k + 1) for _ in range(max_v + 1)]

    if max_k >= 0:
      C_table[0][0] = 1 # Base case: 0 sequences sum to length 0 in 1 way (empty sequence)

    # Base case v=1: C[1][k] = P[k]
    if max_v >= 1:
       for k in range(1, max_k + 1):
           if k < len(P): C_table[1][k] = P[k]
           
    # Compute table using recurrence: C[v][k] = C[v-1][k] - C[v-2][k-1]
    for v in range(2, max_v + 1):
        C_table[v][0] = 0 # C[v][0] is 0 for v >= 1
        for k in range(1, max_k + 1): 
            term1 = C_table[v-1][k]
            term2 = 0
            if k - 1 >= 0: # Ensure index k-1 is valid
               term2 = C_table[v-2][k-1]
            
            C_table[v][k] = (term1 - term2 + mod) % mod


# Fetch precomputed SumCountPartitions(v, k) value from table
def SumCountPartitions(v, k):
    # Basic validity checks for indices v, k
    if v < 0 or k < 0: return 0
    if k == 0: return 1 if v == 0 else 0 # Need v=0 for length 0
    # Minimum length k must be at least v (each of v sequences has length >= 1)
    if k < v: return 0 

    # Check if indices are within precomputed table bounds
    if v < len(C_table) and k < len(C_table[0]):
       return C_table[v][k]
    else:
       # Indices out of bounds. Should not happen with proper precomputation.
       return 0


# Function `solve(idx, k)`: counts Polish sequences W of length k such that W <= A[idx...idx+k-1].
# Uses memoization `memo_solve`.
def solve(idx, k):
    state = (idx, k)
    if state in memo_solve: return memo_solve[state]
    
    # Base cases for length k
    if k <= 0: return 0 # Polish sequences must have positive length
    
    # Boundary check: cannot start sequence if index is out of bounds
    if idx >= N: return 0 

    if k == 1:
        # Only Polish sequence of length 1 is (0). 
        # Since A[idx] >= 0, (0) <= A[idx...] is always true.
        return 1

    # Recursive calculation
    res = 0
    limit_V1 = A[idx] # The value at current index in A determines the upper limit for V1
    
    # Iterate potential first values `v` from 0 up to `limit_V1`.
    for v in range(limit_V1 + 1):
        # Case v = 0: Polish sequence must be (0), length must be 1.
        if v == 0:
            # If k==1, this case corresponds to sequence (0).
            # If A[idx] > 0 (limit_V1 > 0), then (0) < A[idx...], it's a "less than" case. Count 1.
            if k == 1 and limit_V1 > 0:
                # This logic is slightly redundant with base case k=1 handling.
                # The base case k=1 should correctly return 1.
                # Let's refine: The loop should mainly handle recursive step structure.
                # Base case k=1 already handles this. Maybe structure loop differently?
                # Let's keep current structure, ensure correctness.
                # If k=1, base case returns 1. Loop v=0 when limit_V1 >= 0.
                # If limit_V1 = 0, then v=0 is the only iteration. Calls solve_concat -> eventually base cases.
                # If limit_V1 > 0, loop v=0, then v=1..limit_V1.
                # If k=1, v=0, limit_V1 > 0: This is V=(0) < A. Should be counted. Base case handles it?
                # Let's rethink: Base case handles k=1. Loop for v should handle k > 1.
                # If k > 1, v=0 is impossible. Correct to skip using `continue`.
                 pass # Handled by base case k=1 logic returning 1.
            if k > 1: continue # Sequence length > 1 cannot start with 0.

        # Case v > 0: Sequence is (v, W1, ..., Wv). Requires total length k > v.
        if k <= v: continue
            
        # Case 1: v < limit_V1. Strictly less. Tight constraint broken.
        if v < limit_V1:
            # Count ways to form the rest: Sum L(Wi) = k-1. Uses SumCountPartitions.
            res = (res + SumCountPartitions(v, k - 1)) % MOD

        # Case 2: v == limit_V1. Equal. Tight constraint continues for subsequences.
        else: # v == limit_V1
             # Check if A has enough elements for the subsequences W1..Wv of total length k-1
             # Subsequences occupy A[idx+1 ... idx+k-1]. Max index needed is idx+k-1.
             if idx + k <= N: # Ensure ending index idx+k-1 is within 0..N-1
                # Recursive call to count concatenations <= A[idx+1...]
                res = (res + solve_concat(idx + 1, k - 1, v)) % MOD
             # If A too short, count is 0 for this path.

    memo_solve[state] = res
    return res


# Function `solve_concat(idx, k, num_seqs)`: counts ways to form W = W_1...W_{num_seqs}
# where W_i are Polish, sum L(W_i) = k, and concatenated W <= A[idx ... idx+k-1].
# Uses memoization `memo_concat`.
def solve_concat(idx, k, num_seqs):
    state = (idx, k, num_seqs)
    if state in memo_concat: return memo_concat[state]
    
    # Base cases
    if num_seqs == 0:
        return 1 if k == 0 else 0 # Need 0 length for 0 sequences
    if k <= 0: return 0 # Positive length needed if num_seqs > 0
    # Minimum length k must be at least num_seqs
    if k < num_seqs: return 0
    
    # Boundary check for the required range in A: A[idx ... idx+k-1]
    if idx + k > N: return 0 

    res = 0
    # Iterate possible lengths k_1 for the first Polish sequence W_1.
    # Length k_1 must be >= 1.
    # Max length k_1 is such that remaining num_seqs-1 sequences need at least length num_seqs-1.
    # So k_1 <= k - (num_seqs - 1).
    for k_1 in range(1, k - (num_seqs - 1) + 1):
        
        # Count ways to choose W_1 of length k_1 such that W_1 <= A[idx ... idx+k_1-1].
        count_W1_le = solve(idx, k_1)
        if count_W1_le == 0: continue # If no such W_1 possible, skip this k_1.

        # Check if the prefix A[idx...idx+k_1-1] itself is Polish using CheckCache
        is_A_prefix_polish, prefix_len = CheckCache.get(idx, (False, 0))
        # Also check if the Polish prefix found has the exact length k_1
        is_A_prefix_polish = is_A_prefix_polish and (prefix_len == k_1)
            
        # Calculate count for W_1 < A[idx...idx+k_1-1]
        count_W1_lt = count_W1_le
        if is_A_prefix_polish:
            count_W1_lt = (count_W1_le - 1 + MOD) % MOD
            
        # Contribution from W_1 < A[...] case:
        # Need to form remaining num_seqs-1 sequences with total length k-k_1. No tight constraint.
        term1_ways = SumCountPartitions(num_seqs - 1, k - k_1)
        if term1_ways > 0: # Avoid multiplication by 0
           term1 = (count_W1_lt * term1_ways) % MOD
           res = (res + term1) % MOD
            
        # Contribution from W_1 = A[...] case:
        # Possible only if A[idx...idx+k_1-1] is Polish of length k_1.
        if is_A_prefix_polish:
             # Need to form remaining sequences <= A[idx+k_1 ... idx+k-1]. Tight constraint continues.
             term2 = solve_concat(idx + k_1, k - k_1, num_seqs - 1)
             res = (res + term2) % MOD

    memo_concat[state] = res
    return res

# Main execution part
def main():
    global N, A
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Precompute number of Polish sequences of length k
    precompute_P(N, MOD)
    
    # Precompute SumCountPartitions table. This is O(N^2).
    # For N=3e5, this step will likely cause Time Limit Exceeded.
    # This indicates either the approach is wrong for large N, or there's optimization needed.
    # Run it assuming it might pass some cases or potentially reveal structure.
    precompute_SumCountPartitions(N, N, MOD) 

    # Precompute Polish prefix checks for sequence A. O(N).
    precompute_check() 

    # Call the main DP function to solve the problem.
    print(solve(0, N))

if __name__ == '__main__':
    main()