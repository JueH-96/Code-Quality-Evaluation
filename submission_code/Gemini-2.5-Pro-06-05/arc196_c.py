# YOUR CODE HERE
import sys

def solve():
    """
    Solves the strongly connected graph pairing problem.
    """
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    MOD = 998244353

    # Precompute factorials
    MAX_FACT = N + 1
    fact = [1] * MAX_FACT
    for i in range(2, MAX_FACT):
        fact[i] = (fact[i - 1] * i) % MOD

    # Find balanced prefixes
    # w_counts[i] will store the number of 'W's in S[:i]
    w_counts = [0] * (2 * N + 1)
    b_counts = [0] * (2 * N + 1)
    for i in range(2 * N):
        w_counts[i + 1] = w_counts[i] + (1 if S[i] == 'W' else 0)
        b_counts[i + 1] = b_counts[i] + (1 if S[i] == 'B' else 0)

    # A pairing can only be "bad" if there is a prefix i where W(i) = B(i).
    # If W(0) != B(0) (i.e. N!=N), something is wrong with input. W(2N)=B(2N)=N.
    if w_counts[2 * N] != N:
        print(0)
        return

    # zeros are the indices i where the prefix S[:i] is balanced.
    zeros = [i for i in range(1, 2 * N) if w_counts[i] == b_counts[i]]

    # If there are no intermediate balanced prefixes, any pairing is valid.
    if not zeros:
        print(fact[N])
        return

    # Add 2N to handle the full problem in the DP
    zeros.append(2 * N)

    # dp[i] = number of valid pairings for the subproblem on vertices {1, ..., zeros[i-1]}
    # The size of dp will be len(zeros) + 1. dp[0] is for the trivial empty prefix.
    dp = [0] * (len(zeros) + 1)
    dp[0] = 1
    
    # Store W-counts at zero points for convenience
    w_at_zeros = [0] + [w_counts[z] for z in zeros]
    
    # DP using Inclusion-Exclusion principle
    for i in range(1, len(zeros) + 1):
        # Total ways to pair in prefix up to zeros[i-1]
        total_ways = fact[w_at_zeros[i]]
        
        # Subtract invalid ways. An invalid way respects a smaller partition j < i.
        invalid_ways = 0
        for j in range(1, i):
            # A valid pairing up to zeros[j-1] (dp[j])
            # followed by any pairing for the segment between zeros[j-1] and zeros[i-1].
            # Number of pairs in this segment is w_at_zeros[i] - w_at_zeros[j].
            num_pairs_in_segment = w_at_zeros[i] - w_at_zeros[j]
            ways_for_segment = fact[num_pairs_in_segment]
            
            term = (dp[j] * ways_for_segment) % MOD
            invalid_ways = (invalid_ways + term) % MOD
            
        dp[i] = (total_ways - invalid_ways + MOD) % MOD
        
    print(dp[-1])

solve()