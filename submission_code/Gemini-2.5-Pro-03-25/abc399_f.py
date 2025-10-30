# YOUR CODE HERE
import sys

# Set higher recursion depth for safety, although not strictly needed for this iterative solution
# sys.setrecursionlimit(200000) 

def solve():
    # Read N and K from input
    N, K = map(int, sys.stdin.readline().split())
    # Read the sequence A
    A = list(map(int, sys.stdin.readline().split()))
    
    # Define the modulus
    MOD = 998244353

    # Compute prefix sums P.
    # P[i] stores the sum A[0] + ... + A[i-1] modulo MOD.
    # P has length N+1. P[0] is initialized to 0.
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = (P[i] + A[i]) % MOD

    # Precompute binomial coefficients C(K, j) % MOD for j = 0 to K.
    # We use the formula C(n, k) = C(n, k-1) * (n - k + 1) / k.
    # This requires modular inverse for division.
    # C has length K+1. C[j] will store C(K, j) % MOD.
    C = [0] * (K + 1)
    C[0] = 1  # C(K, 0) = 1
    
    # Compute C(K, i) iteratively using C(K, i-1)
    for i in range(1, K + 1):
        # Calculate C(K, i) = C(K, i-1) * (K - i + 1) * i^(-1) mod MOD
        
        # Multiply by (K - i + 1)
        C[i] = (C[i-1] * (K - i + 1)) % MOD
        
        # Compute modular inverse of i using Fermat's Little Theorem: i^(MOD-2) % MOD
        # This is valid because MOD is prime and i <= K <= 10, so i is not divisible by MOD.
        inv_i = pow(i, MOD - 2, MOD)
        
        # Multiply by the modular inverse of i (equivalent to division)
        C[i] = (C[i] * inv_i) % MOD

    # Initialize the total sum we want to compute
    total_sum = 0
    
    # Q_k_sums[k] will dynamically store the sum sum_{i=0}^{r-1} P[i]^k % MOD.
    # This array represents the necessary state carried over from previous iterations.
    # It has size K+1, storing sums for powers k=0 to K.
    Q_k_sums = [0] * (K + 1)
    
    # Initialize Q_k_sums for the first iteration where r=1.
    # The sum required is over i=0 to r-1=0. So we just need P[0]^k.
    # Since P[0] = 0, by convention 0^0 = 1, and P[0]^k = 0 for k > 0.
    # Thus, Q_k_sums initialized represents [P[0]^0, P[0]^1, ..., P[0]^K].
    Q_k_sums[0] = 1 
    
    # current_P_powers[k] will temporarily store P[r]^k % MOD during iteration r.
    current_P_powers = [0] * (K + 1)

    # Iterate through r from 1 to N. Each 'r' corresponds to the right endpoint of subarrays (1-based index).
    for r in range(1, N + 1):
        
        # Compute powers P[r]^k for k = 0..K efficiently.
        current_P_val = P[r]  # Current prefix sum value P_r
        current_P_powers[0] = 1 # P[r]^0 = 1 (holds even if P[r]=0)
        for k in range(1, K + 1):
            # Compute P[r]^k = P[r]^(k-1) * P[r] using the previously computed power
            current_P_powers[k] = (current_P_powers[k-1] * current_P_val) % MOD

        # Compute the contribution of this 'r' to the total sum.
        # The problem asks for Sum_{1<=l<=r<=N} (Sum_{i=l..r} A_i)^K mod MOD.
        # This is Sum_{r=1..N} Sum_{l=1..r} (P[r] - P[l-1])^K mod MOD.
        # Using binomial expansion and changing summation order, we arrive at the formula:
        # Contribution(r) = sum_{j=0}^K C(K, j) * (-1)^(K-j) * P[r]^j * (sum_{i=0}^{r-1} P[i]^(K-j))
        # The inner sum (sum_{i=0}^{r-1} P[i]^(K-j)) is stored in Q_k_sums[K-j].
        
        r_contribution = 0
        for j in range(K + 1):
            # Calculate the term: C(K, j) * P[r]^j * Q_k_sums[K-j]
            term = (C[j] * current_P_powers[j]) % MOD
            term = (term * Q_k_sums[K-j]) % MOD
            
            # Apply the sign (-1)^(K-j). If K-j is odd, the term is negated modulo MOD.
            if (K - j) % 2 == 1: 
                # Modular negation: (MOD - term) % MOD correctly handles negative results
                term = (MOD - term) % MOD
            
            # Accumulate the term to r_contribution for the current r
            r_contribution = (r_contribution + term) % MOD
        
        # Add this r's contribution to the overall total sum
        total_sum = (total_sum + r_contribution) % MOD

        # Update the Q_k_sums array for the next iteration (r+1).
        # The sum needs to include terms for i=r.
        # Updated Q_k_sums[k] will be sum_{i=0}^{r} P[i]^k
        # This is achieved by adding P[r]^k (which is current_P_powers[k]) to the current Q_k_sums[k].
        for k in range(K + 1):
            Q_k_sums[k] = (Q_k_sums[k] + current_P_powers[k]) % MOD
            
    # Print the final computed total sum modulo MOD
    print(total_sum)

# Execute the solve function to run the program
solve()