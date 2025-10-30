import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MOD = 998244353

    # ans[k] will store the number of arithmetic subsequences of length k.
    # Using 1-based indexing for length, so ans array is size N+1, ans[0] is unused.
    ans = [0] * (N + 1)

    if N == 0: # Problem constraints: N >= 1. This case won't be hit.
        print("")
        return
        
    # All single elements are arithmetic subsequences of length 1.
    ans[1] = N

    # dp[i] is a dictionary. Keys are differences `d`.
    # dp[i][d] is a list where dp[i][d][k] is the count of
    # arithmetic subsequences ending at index `i` (with value A[i]),
    # having common difference `d`, and of length `k`.
    # Lengths `k` are from 2 to N.
    # The list dp[i][d] has size N+1 (indices 0 to N), effectively 1-indexed for length.
    dp = [defaultdict(lambda: [0] * (N + 1)) for _ in range(N)]

    # Iterate over each element A[i] as the potential end of an arithmetic subsequence.
    for i in range(N):
        # Iterate over previous elements A[j] as the potential second-to-last element.
        for j in range(i):
            diff = A[i] - A[j]
            
            # Case 1: The subsequence (A[j], A[i]) is an arithmetic subsequence of length 2.
            # It ends at A[i] and has common difference `diff`.
            # Increment count for length 2 APs ending at A[i] with this `diff`.
            dp[i][diff][2] = (dp[i][diff][2] + 1) % MOD
            
            # Add this AP of length 2 to the total count for length 2 subsequences.
            ans[2] = (ans[2] + 1) % MOD
            
            # Case 2: Extend arithmetic subsequences that ended at A[j] with the same common difference `diff`.
            # An AP (..., A_k, A_j) ending at A[j] of length `prev_len` (where `prev_len >= 2`) 
            # can be extended by A[i] to form a new AP (..., A_k, A_j, A_i) of length `prev_len + 1`.
            
            # The maximum possible length of an AP ending at A[j] is j+1.
            # So, `prev_len` can range from 2 up to j+1.
            # The loop `range(2, j + 2)` covers `prev_len` values: 2, 3, ..., j+1.
            for prev_len in range(2, j + 2): 
                if dp[j][diff][prev_len] > 0:
                    count_prev_ap = dp[j][diff][prev_len]
                    current_len = prev_len + 1
                    
                    # Add these extended APs to dp state for A[i].
                    dp[i][diff][current_len] = (dp[i][diff][current_len] + count_prev_ap) % MOD
                    # Add to total count for APs of length `current_len`.
                    ans[current_len] = (ans[current_len] + count_prev_ap) % MOD
                        
    # Print answers for lengths 1 to N, space-separated.
    # ans[0] is unused, so slice from ans[1].
    print(*(ans[1:]))

if __name__ == '__main__':
    solve()