import sys

def main():
    N = int(sys.stdin.readline())
    garbage_params = []
    for _ in range(N):
        q, r = map(int, sys.stdin.readline().split())
        garbage_params.append((q, r))

    Q = int(sys.stdin.readline())
    
    output_lines = []

    for _ in range(Q):
        t_query, d_query = map(int, sys.stdin.readline().split())
        
        # t_query is 1-indexed, so adjust to 0-indexed for list access
        q_val, r_val = garbage_params[t_query - 1]
        
        # We are looking for the smallest collection day D_ans such that D_ans >= d_query.
        # Collection days are of the form K_actual * q_val + r_val, 
        # where K_actual is a non-negative integer (K_actual >= 0).
        
        # So we need to find the smallest integer K_actual >= 0 such that:
        # K_actual * q_val + r_val >= d_query
        
        # Rearranging the inequality:
        # K_actual * q_val >= d_query - r_val
        
        # Let X_val = d_query - r_val.
        # We need K_actual * q_val >= X_val.
        # Since q_val > 0 (guaranteed by 0 <= r_i < q_i, so q_i >= 1), we can divide by q_val:
        # K_actual >= X_val / q_val.
        
        # Since K_actual must be an integer, K_actual must be at least ceil(X_val / q_val).
        # Let K_candidate_from_ceil = ceil(X_val / q_val).
        # The ceiling function ceil(A/B) for an integer A and positive integer B 
        # can be computed as (A + B - 1) // B using integer division that floors 
        # (which is Python's // operator).
        X_val = d_query - r_val
        K_candidate_from_ceil = (X_val + q_val - 1) // q_val
        
        # We also have the constraint that K_actual must be non-negative (K_actual >= 0).
        # So, the smallest valid K_actual is max(0, K_candidate_from_ceil).
        K_final = max(0, K_candidate_from_ceil)
        
        # The next collection day is then K_final * q_val + r_val.
        ans = K_final * q_val + r_val
        output_lines.append(str(ans))
        
    sys.stdout.write("
".join(output_lines) + "
")

if __name__ == '__main__':
    main()