import sys
from collections import Counter

def solve():
    # Read N and T from the first line
    first_line = sys.stdin.readline().split()
    N = int(first_line[0])
    T = first_line[1]

    # Read S_i strings
    S = []
    for _ in range(N):
        S.append(sys.stdin.readline().strip())

    len_T = len(T)
    T_rev = T[::-1] # Precompute reversed T for R_values calculation

    P_values = [0] * N # Stores P_x for each S_x
    R_values = [0] * N # Stores R_x for each S_x

    # --- Step 2: Calculate P_values and R_values ---

    # Calculate P_values (length of longest prefix of T that is a subsequence of S_x)
    for i in range(N):
        current_s_char_idx = 0 # Pointer for S[i]
        current_t_char_idx = 0 # Pointer for T (length of matched prefix of T)
        
        while current_s_char_idx < len(S[i]) and current_t_char_idx < len_T:
            if S[i][current_s_char_idx] == T[current_t_char_idx]:
                current_t_char_idx += 1 # Match found, advance T pointer
            current_s_char_idx += 1 # Always advance S[i] pointer
        P_values[i] = current_t_char_idx

    # Calculate R_values (length of longest suffix of T that is a subsequence of S_x)
    # This is equivalent to finding the longest prefix of T_rev in S_x_rev
    for i in range(N):
        current_s_char_idx = len(S[i]) - 1 # Pointer for S[i], starting from end
        current_t_char_idx_rev = 0 # Pointer for T_rev (length of matched prefix of T_rev)
        
        while current_s_char_idx >= 0 and current_t_char_idx_rev < len_T:
            if S[i][current_s_char_idx] == T_rev[current_t_char_idx_rev]:
                current_t_char_idx_rev += 1 # Match found, advance T_rev pointer
            current_s_char_idx -= 1 # Always advance S[i] pointer backwards
        R_values[i] = current_t_char_idx_rev

    # --- Step 3: Count pairs (i, j) such that P_values[i] + R_values[j] >= len_T ---

    total_pairs = 0

    # Use a frequency counter for R_values to efficiently query counts
    R_freq = Counter(R_values)
    
    # Create suffix sums for R_freq
    # R_suffix_sum[k] will store the count of S_x such that R_x >= k
    # Size len_T + 2 to handle indices up to len_T + 1
    R_suffix_sum = [0] * (len_T + 2) 
    
    # Populate R_suffix_sum from right to left
    for k in range(len_T, -1, -1):
        R_suffix_sum[k] = R_freq[k] + R_suffix_sum[k+1]

    # Iterate through each P_value and sum up corresponding R_values counts
    for p_val in P_values:
        # We need R_j such that R_j >= len_T - p_val
        # Calculate the minimum R_j value required
        needed_r_val = len_T - p_val
        
        # If needed_r_val is 0 or negative (meaning p_val >= len_T), then S_i itself contains T.
        # Any S_j will satisfy the condition. R_suffix_sum[0] counts all N strings.
        # So using max(0, needed_r_val) handles this case correctly.
        # The index for R_suffix_sum will always be within [0, len_T] since p_val is in [0, len_T].
        total_pairs += R_suffix_sum[max(0, needed_r_val)]

    # Print the final answer
    sys.stdout.write(str(total_pairs) + "
")

# Call the solve function
solve()