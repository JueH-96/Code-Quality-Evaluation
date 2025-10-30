import sys

def main():
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    if N <= 1:
        for _ in range(Q):
            # Must consume input lines for l, r even if N <= 1
            sys.stdin.readline() 
            sys.stdout.write("0
")
        return

    # P_values[k] stores the count of indices j such that 0 <= j <= k
    # and S[j] == S[j+1] (using 0-indexed S and j).
    # P_values array has N-1 elements, indexed from 0 to N-2.
    P_values = [0] * (N - 1) 
    
    if S[0] == S[1]:
        P_values[0] = 1
    
    # k ranges from 1 up to N-2.
    # S[k] refers to the (k+1)-th character of the original 1-indexed string.
    # The pair S[k]S[k+1] corresponds to S_{k+1}S_{k+2} in 1-indexing.
    for k in range(1, N - 1): 
        P_values[k] = P_values[k-1] 
        if S[k] == S[k+1]:
            P_values[k] += 1
            
    ans_lines = []
    for _ in range(Q):
        l_orig, r_orig = map(int, sys.stdin.readline().split()) # 1-indexed query
        
        # Problem asks for count of 1-indexed p such that:
        # l_orig <= p <= r_orig-1 AND S_p = S_{p+1}
        #
        # In 0-indexed terms (let p' = p-1):
        # (l_orig-1) <= p' <= (r_orig-1)-1  AND S[p'] == S[p'+1]
        # So, p' ranges from l_orig-1 to r_orig-2.
        #
        # We need sum of B[k] for k from (l_orig-1) to (r_orig-2).
        # Using P_values, this is P_values[r_orig-2] - P_values[(l_orig-1)-1]
        # which is P_values[r_orig-2] - P_values[l_orig-2].

        idx_sum_end = r_orig - 2
        idx_sum_start_minus_1 = l_orig - 2
        
        val_at_idx_sum_end = 0
        # Check if idx_sum_end is a valid index for P_values
        if idx_sum_end >= 0: 
            val_at_idx_sum_end = P_values[idx_sum_end]
        
        val_at_idx_sum_start_minus_1 = 0
        # Check if idx_sum_start_minus_1 is a valid index for P_values
        if idx_sum_start_minus_1 >= 0: 
            val_at_idx_sum_start_minus_1 = P_values[idx_sum_start_minus_1]
            
        current_ans = val_at_idx_sum_end - val_at_idx_sum_start_minus_1
        ans_lines.append(str(current_ans))
        
    sys.stdout.write("
".join(ans_lines) + "
")

if __name__ == '__main__':
    main()