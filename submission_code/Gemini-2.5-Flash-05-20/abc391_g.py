import sys
from collections import defaultdict

def solve():
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    MOD = 998244353

    # dp[mask] stores the count of strings T of the current length
    # that result in the LCS profile represented by 'mask'.
    # A mask is an integer where the i-th bit (0-indexed, i from 0 to N-1)
    # corresponds to the difference l_{i+1} - l_i for the LCS profile (l_0, ..., l_N).
    # l_0 is always 0. l_k = sum_{j=1 to k} (mask >> (j-1)) & 1.
    dp = defaultdict(int)
    # Initial state: empty string T (length 0).
    # LCS profile (l_0, ..., l_N) is all zeros (0, 0, ..., 0).
    # This corresponds to mask 0.
    dp[0] = 1

    # Iterate M times, building strings of length 1 to M.
    # Each iteration corresponds to appending one character.
    for _ in range(M):
        new_dp = defaultdict(int)
        for prev_mask, count in dp.items():
            if count == 0:
                continue

            # Reconstruct the l_prev profile (LCS_vals for S[:i] with T_prev)
            # l_prev[i] is LCS(S[:i], T_prev_prefix)
            l_prev = [0] * (N + 1)
            for i in range(1, N + 1):
                # l_prev[i] = l_prev[i-1] + (1 if (i-1)-th bit of prev_mask is set else 0)
                l_prev[i] = l_prev[i-1] + ((prev_mask >> (i-1)) & 1)

            # Iterate over all 26 possible characters to append
            for char_idx in range(26):
                c = chr(ord('a') + char_idx)
                
                # Compute the l_curr profile (LCS_vals for S[:i] with T_curr)
                l_curr = [0] * (N + 1)
                curr_mask = 0
                for i in range(1, N + 1):
                    # Standard LCS recurrence:
                    # LCS(S[:i], T_curr) based on LCS(S[:i-1], T_prev) and LCS(S[:i], T_prev)
                    # S[i-1] is the i-th character of S (1-indexed based on l_i)
                    if S[i-1] == c:
                        l_curr[i] = l_prev[i-1] + 1
                    else:
                        l_curr[i] = max(l_curr[i-1], l_prev[i])
                    
                    # Construct curr_mask from l_curr profile
                    # If l_curr[i] > l_curr[i-1], it means l_curr[i] = l_curr[i-1] + 1,
                    # so the (i-1)-th bit of curr_mask should be set.
                    if l_curr[i] > l_curr[i-1]:
                        curr_mask |= (1 << (i-1))
                
                # Add the count from prev_mask to the new_dp entry for curr_mask
                new_dp[curr_mask] = (new_dp[curr_mask] + count) % MOD
        
        # Move to the next iteration with the newly computed DP table
        dp = new_dp

    # After M iterations, dp contains counts for strings of length M.
    # Now aggregate results by the final LCS length (LCS_vals[N]).
    ans = [0] * (N + 1) # ans[k] will store the count for LCS length k

    for final_mask, count in dp.items():
        # The LCS length for S and T (LCS_vals[N]) is the sum of all N differences.
        # This is equivalent to summing the set bits in the final_mask.
        l_N_val = 0
        for i in range(N): # Iterate through bits 0 to N-1
            l_N_val += ((final_mask >> i) & 1)
        
        ans[l_N_val] = (ans[l_N_val] + count) % MOD

    # Print the results
    sys.stdout.write(" ".join(map(str, ans)) + "
")

if __name__ == '__main__':
    solve()