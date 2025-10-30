import sys

# Function to solve the problem
def solve():
    # Read N and K from standard input
    N, K = map(int, sys.stdin.readline().split())
    # Read the sequence A from standard input
    A = list(map(int, sys.stdin.readline().split()))
    
    # Modulo value
    MOD = 998244353

    # 1. Compute prefix sums P_j
    # prefix_sum[j] = sum_{i=1 to j} A_i
    # prefix_sum[0] = 0
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = (prefix_sum[i] + A[i]) % MOD

    # 2. Compute powers of prefix sums P_j^k
    # P_pow[j][k] = prefix_sum[j]^k mod MOD
    # P_pow[j][0] = prefix_sum[j]^0 = 1 (even for prefix_sum[j] == 0)
    # P_pow[j][k] = P_pow[j][k-1] * prefix_sum[j] mod MOD
    P_pow = [[0] * (K + 1) for _ in range(N + 1)]
    for j in range(N + 1):
        P_pow[j][0] = 1 # Base case: power 0 is always 1
        for k in range(1, K + 1):
            P_pow[j][k] = (P_pow[j][k - 1] * prefix_sum[j]) % MOD

    # 3. Compute binomial coefficients C(K, m)
    # binom[k][m] = C(k, m) mod MOD
    # Using Pascal's identity: C(k, m) = C(k-1, m-1) + C(k-1, m)
    # Base cases: C(k, 0) = 1, C(k, k) = 1
    binom = [[0] * (K + 1) for _ in range(K + 1)]
    for k in range(K + 1):
        binom[k][0] = 1
        if k <= K: # We only need coefficients up to row K
            for m in range(1, k + 1):
                binom[k][m] = (binom[k - 1][m - 1] + binom[k - 1][m]) % MOD

    # 4. Compute the total sum
    # The sum is sum_{1 <= l <= r <= N} (sum_{i=l to r} A_i)^K
    # Let S(l, r) = sum_{i=l to r} A_i = P_r - P_{l-1}
    # Sum = sum_{r=1 to N} sum_{l=1 to r} (P_r - P_{l-1})^K
    # Let j = l-1, so 0 <= j <= r-1
    # Sum = sum_{r=1 to N} sum_{j=0 to r-1} (P_r - P_j)^K
    # Using binomial expansion: (P_r - P_j)^K = sum_{m=0 to K} C(K, m) * P_r^m * (-P_j)^{K-m}
    # Sum = sum_{r=1 to N} sum_{j=0 to r-1} sum_{m=0 to K} C(K, m) * P_r^m * (-1)^{K-m} * P_j^{K-m}
    # Swap summation order:
    # Sum = sum_{m=0 to K} C(K, m) * (-1)^{K-m} * sum_{r=1 to N} P_r^m * (sum_{j=0 to r-1} P_j^{K-m})
    # Let Q_k(r) = sum_{j=0 to r-1} P_j^k.
    # Sum = sum_{m=0 to K} C(K, m) * (-1)^{K-m} * sum_{r=1 to N} P_r^m * Q_{K-m}(r)

    total_sum = 0

    # Iterate over m from 0 to K
    for m in range(K + 1):
        k_val = K - m # This is the power for P_j in Q function

        # Compute the inner sum: sum_{r=1 to N} P_r^m * Q_{k_val}(r)
        # Q_k_val(r) = sum_{j=0 to r-1} P_j^k_val
        # We compute Q_k_val(r) iteratively while iterating r
        # Q_k_val(r) = Q_k_val(r-1) + P_{r-1}^k_val for r >= 1, with Q_k_val(0) = 0

        current_Q_k_val = 0 # This variable will hold Q_k_val(r) for the current r
                            # It is initialized to Q_k_val(0) = sum_{j=0}^{-1} P_j^k_val = 0

        sum_over_r = 0      # This variable will hold sum_{r=1 to N} P_r^m * Q_{k_val}(r)

        # Iterate over r from 1 to N
        for r in range(1, N + 1):
            # Update current_Q_k_val from Q_k_val(r-1) to Q_k_val(r).
            # Q_k_val(r) = Q_k_val(r-1) + P_{r-1}^k_val.
            # At the start of iteration r, `current_Q_k_val` holds Q_k_val(r-1) = sum_{j=0}^{r-2} P_j^{k_val}.
            # We need to add P_{r-1}^{k_val} to get Q_k_val(r).
            current_Q_k_val = (current_Q_k_val + P_pow[r - 1][k_val]) % MOD

            # Now `current_Q_k_val` holds Q_k_val(r) = sum_{j=0 to r-1} P_j^k_val.
            # Calculate the term for the current r: P_r^m * Q_{k_val}(r)
            term_r = (P_pow[r][m] * current_Q_k_val) % MOD

            # Add the term to the inner sum
            sum_over_r = (sum_over_r + term_r) % MOD

        # After the inner loop over r, sum_over_r holds sum_{r=1 to N} P_r^m * Q_{k_val}(r)

        # Add the term for the current m to the total sum
        # The term is C(K, m) * (-1)^(K-m) * sum_over_r
        term_m = (binom[K][m] * sum_over_r) % MOD

        # Determine the sign of (-1)^(K-m)
        sign = 1 if k_val % 2 == 0 else -1

        # Add or subtract the term based on the sign
        if sign == 1:
            total_sum = (total_sum + term_m) % MOD
        else:
            # Add MOD before taking modulo to handle negative results correctly
            total_sum = (total_sum - term_m + MOD) % MOD

    # Print the final result
    print(total_sum)

# Call the solve function to run the program
solve()