import sys

def main():
    N = int(sys.stdin.readline())
    S_list = [sys.stdin.readline().strip() for _ in range(N)]

    # Calculate M, the maximum length of the strings S_i
    # Constraints: N >= 1, len(S_i) >= 1. So S_list is not empty and all lengths are >= 1.
    # Thus, M will be >= 1.
    M = 0
    if N > 0: # This check is technically redundant due to N >= 1 constraint
        for s_val in S_list:
            if len(s_val) > M:
                M = len(s_val)
    # A more Pythonic way for M: M = max(len(s) for s in S_list) if S_list else 0
    # Given constraints, M = max(len(s) for s in S_list) is sufficient.


    output_T_strings = []

    # j_char_idx iterates from 0 to M-1.
    # This index corresponds to:
    #   - The (j_char_idx)-th character (0-indexed) to be picked from each S_i.
    #   - Building the T_{j_char_idx+1} string.
    for j_char_idx in range(M):
        
        current_T_char_list = []
        
        # p_in_T iterates from 1 to N (1-based position of character in T string).
        # The p_in_T-th character of the current T string is derived from S_{N - p_in_T + 1}.
        for p_in_T in range(1, N + 1):
            
            # Convert 1-based S-index (N - p_in_T + 1) to 0-based S_list index.
            # S_list_idx = (N - p_in_T + 1) - 1 = N - p_in_T.
            s_list_idx_for_current_char = N - p_in_T
            
            target_S_string = S_list[s_list_idx_for_current_char]
            
            # Check if target_S_string has a character at j_char_idx
            if j_char_idx < len(target_S_string):
                char_to_add = target_S_string[j_char_idx]
            else:
                # target_S_string is shorter than j_char_idx+1 characters.
                char_to_add = '*'
            
            current_T_char_list.append(char_to_add)
        
        # Form the raw T string for the current j_char_idx (T_{j_char_idx+1})
        # Characters are appended in order: from S_N, S_{N-1}, ..., S_1
        raw_T_string = "".join(current_T_char_list)
        
        # Trim trailing '*' characters as per problem condition
        trimmed_T_string = raw_T_string.rstrip('*')
        
        output_T_strings.append(trimmed_T_string)

    # Print the resulting T strings
    for t_str in output_T_strings:
        sys.stdout.write(t_str + "
")

if __name__ == '__main__':
    main()