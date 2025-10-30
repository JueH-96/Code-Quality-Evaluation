import sys

def power(a, b, mod):
    res = 1
    a %= mod
    while b > 0:
        if b % 2 == 1:
            res = (res * a) % mod
        a = (a * a) % mod
        b //= 2
    return res

def modInverse(n, mod):
    return power(n, mod - 2, mod)

# M is the target sum
M = 10
MOD = 998244353

def solve():
    # Read input
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # DP state: dp[mask] = number of outcomes for the first i dice
    # such that the set of achievable subset sums <= M is represented by mask.
    # We use only two rows for DP (current and next).
    dp = [0] * (1 << (M + 1))
    dp[1] = 1 # Mask 1 represents the set {0} initially achievable

    # Precompute mask for clearing bits > M
    mask_upper = (1 << (M + 1)) - 1

    for i in range(N):
        A_i = A[i]
        next_dp = [0] * (1 << (M + 1))

        # Iterate through all possible outcomes r for the current die
        # r takes values from 1 to A_i
        for r in range(1, A_i + 1):
            for old_mask in range(1 << (M + 1)):
                if dp[old_mask] == 0:
                    continue

                # Calculate new_mask
                # S_old is the set of achievable sums <= M represented by old_mask
                # The new set of achievable sums is S_old U {s + r | s in S_old}
                # We only care about sums <= M.
                # The mask for {s + r | s in S_old} is conceptually (old_mask << r)
                # We filter sums s+r that are <= M. This is achieved by bitwise AND with mask_upper.
                # The new mask represents the union of sums in S_old and new sums <= M.
                
                new_mask = old_mask | ((old_mask << r) & mask_upper)
                
                next_dp[new_mask] = (next_dp[new_mask] + dp[old_mask]) % MOD

        dp = next_dp # Move to the next DP state

    # Total number of outcomes is product of A_i modulo MOD
    total_outcomes = 1
    for val_A in A:
        total_outcomes = (total_outcomes * val_A) % MOD

    # Number of outcomes where no subset sum is M
    # This is the sum of dp[mask] for masks where the M-th bit is 0
    num_no_sum_M = 0
    for mask in range(1 << (M + 1)):
        if (mask >> M) & 1 == 0: # Check if M-th bit is 0 (sum M is not achievable <= M)
            num_no_sum_M = (num_no_sum_M + dp[mask]) % MOD

    # Probability of no subset sum is M
    # Prob_no_sum_M = num_no_sum_M / total_outcomes (mod MOD)
    prob_no_sum_M = (num_no_sum_M * modInverse(total_outcomes, MOD)) % MOD

    # Probability of at least one subset sum is M
    prob_sum_M = (1 - prob_no_sum_M + MOD) % MOD

    print(prob_sum_M)

solve()