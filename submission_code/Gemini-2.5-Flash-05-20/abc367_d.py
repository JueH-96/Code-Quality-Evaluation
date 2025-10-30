import sys
from collections import Counter

# It's good practice to encapsulate the logic in a function
def solve():
    # Read N and M from the first line of input
    N, M = map(int, sys.stdin.readline().split())
    # Read the N integers for A from the second line of input
    A = list(map(int, sys.stdin.readline().split()))

    # P[k] will store the cumulative sum of steps A_0 + ... + A_{k-1}.
    # P[0] is 0 (representing the start point or "before" the first area).
    # P[N] will store the total circumference (sum of all A_i).
    P = [0] * (N + 1)
    current_sum = 0
    for i in range(N):
        current_sum += A[i]
        P[i+1] = current_sum

    # mods[k] will store P[k] % M.
    # mods[0] = 0 % M = 0.
    # mods[N] = (Total sum of A_i) % M, which is total_sum_mod_M.
    mods = [p % M for p in P]

    total_sum_mod_M = mods[N]

    total_possible_pairs = 0

    # Case 1: s < t (clockwise path does not cross rest area 1 via N -> 1)
    # The length is P[t-1] - P[s-1]. We need (P[t-1] - P[s-1]) % M == 0.
    # This means mods[t-1] == mods[s-1].
    # Here, s and t are 1-indexed. The corresponding 0-indexed positions in mods are s_idx = s-1 and t_idx = t-1.
    # We iterate k from 0 to N-1 (representing t_idx). For each k, we look for s_idx < k such that mods[s_idx] == mods[k].
    freq_map_case1 = Counter()
    for k in range(N): # k ranges from 0 to N-1, covering mods[0] through mods[N-1]
        val_at_k = mods[k]
        # Add the count of elements seen so far that have the same modulo value
        total_possible_pairs += freq_map_case1[val_at_k]
        # Increment the count for the current modulo value
        freq_map_case1[val_at_k] += 1

    # Case 2: s > t (clockwise path crosses rest area 1 via N -> 1)
    # The length is (P[N] - P[s-1]) + P[t-1]. We need this to be a multiple of M.
    # This means (mods[N] - mods[s-1] + mods[t-1]) % M == 0.
    # Rearranging: (mods[t-1] + mods[N]) % M == mods[s-1].
    # We iterate k from 0 to N-1 (representing s_idx). For each k, we look for t_idx < k such that
    # (mods[t_idx] + total_sum_mod_M) % M == mods[k].
    # The value we are searching for in the frequency map is (mods[k] - total_sum_mod_M + M) % M.
    freq_map_case2 = Counter()
    for k in range(N): # k ranges from 0 to N-1, covering mods[0] through mods[N-1]
        current_s_val = mods[k]
        # Calculate the target value that mods[t_idx] should have
        # Adding M before modulo handles negative results from subtraction
        target_t_val = (current_s_val - total_sum_mod_M + M) % M
        # Add the count of elements seen so far that match the target_t_val
        total_possible_pairs += freq_map_case2[target_t_val]
        # Increment the count for the current value (mods[k]) as it might be a t_idx for future s_idx's
        freq_map_case2[current_s_val] += 1

    # Print the final answer
    sys.stdout.write(str(total_possible_pairs) + "
")

# Call the solve function to execute the program
solve()