import sys

# Helper for find_leftmost in S_list[from_idx: to_idx+1] for char < X_char
# Returns -1 if not found, otherwise the index in S_list.
def find_leftmost_in_s_list(s_list_ref, start_idx, end_idx, x_char):
    if start_idx > end_idx : # Empty range
        return -1
    for i in range(start_idx, end_idx + 1):
        if s_list_ref[i] < x_char:
            return i
    return -1

# Helper for find_rightmost in S_list[from_idx: to_idx+1] for char < X_char
# Returns -1 if not found, otherwise the index in S_list.
def find_rightmost_in_s_list(s_list_ref, start_idx, end_idx, x_char):
    if start_idx > end_idx: # Empty range
        return -1
    for i in range(end_idx, start_idx - 1, -1):
        if s_list_ref[i] < x_char:
            return i
    return -1

def solve():
    N, M = map(int, sys.stdin.readline().split())
    S_str = sys.stdin.readline().strip()
    T_str = sys.stdin.readline().strip()

    S_list = list(S_str)
    T_list = list(T_str)

    R = ['0'] * M 
    if M > 0:
        R[M-1] = T_list[M-1]
        for k_idx in range(M-2, -1, -1):
            R[k_idx] = max(T_list[k_idx], R[k_idx+1])

    for k in range(M):
        current_T_char = T_list[k]
        future_best_T_char = R[k]
        
        chosen_idx_for_T_k = -1

        if current_T_char == future_best_T_char:
            # Case (a)
            j_star = find_leftmost_in_s_list(S_list, 0, N-1, current_T_char)
            if j_star != -1:
                chosen_idx_for_T_k = j_star
            else:
                chosen_idx_for_T_k = N-1
        else:
            # Case (b) current_T_char < future_best_T_char
            j0 = find_leftmost_in_s_list(S_list, 0, N-1, future_best_T_char)
            
            j_potential_candidate = -1

            if j0 != -1: # j0 exists, a position is "reserved"
                # Try rightmost j < j0
                j_cand1 = find_rightmost_in_s_list(S_list, 0, j0 - 1, current_T_char)
                
                # Try rightmost j > j0
                j_cand2 = find_rightmost_in_s_list(S_list, j0 + 1, N - 1, current_T_char)
                
                if j_cand1 != -1 and j_cand2 != -1:
                    j_potential_candidate = max(j_cand1, j_cand2) 
                elif j_cand1 != -1:
                    j_potential_candidate = j_cand1
                elif j_cand2 != -1:
                    j_potential_candidate = j_cand2
                # else j_potential_candidate remains -1
            else: # j0 == -1 (future_best_T_char cannot improve S, so no position is "reserved")
                j_potential_candidate = find_rightmost_in_s_list(S_list, 0, N - 1, current_T_char)
            
            if j_potential_candidate != -1:
                chosen_idx_for_T_k = j_potential_candidate
            else:
                # No j != j0 (or no j at all if j0 was -1) could be improved by current_T_char
                # Now consider j0 (if it exists and is valid, i.e. < N)
                if j0 != -1 and S_list[j0] < current_T_char: # Check j0 itself
                    chosen_idx_for_T_k = j0
                else:
                    # Fallback: place current_T_char at S[N-1]
                    chosen_idx_for_T_k = N-1
        
        S_list[chosen_idx_for_T_k] = current_T_char

    sys.stdout.write("".join(S_list) + "
")

solve()