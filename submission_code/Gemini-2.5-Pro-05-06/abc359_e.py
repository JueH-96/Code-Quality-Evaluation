import sys

def main():
    N = int(sys.stdin.readline())
    # H_in[i] stores H_{i+1} from the problem statement.
    # e.g., H_in[0] is H_1, H_in[N-1] is H_N.
    H_in = list(map(int, sys.stdin.readline().split()))

    # A[i] stores A_i from the problem statement.
    # Array A has N+1 elements: A_0, A_1, ..., A_N.
    A = [0] * (N + 1) 
    
    # ans_values[i] stores the answer for A_{i+1} > 0.
    # e.g., ans_values[0] is for A_1, ans_values[N-1] is for A_N.
    ans_values = [0] * N 
    
    num_ans_found = 0
    op_count = 0
    
    while num_ans_found < N:
        op_count += 1
        A[0] += 1

        # This loop corresponds to "For i=1, 2, ..., N in this order" from problem.
        # Let k_prob be the 'i' from problem statement. k_prob ranges from 1 to N.
        for k_prob in range(1, N + 1):
            # The transfer is from A_{k_prob-1} to A_{k_prob}.
            # The condition involves H_{k_prob}.
            
            idx_from = k_prob - 1
            idx_to = k_prob
            
            # H_{k_prob} is H_in[k_prob-1] in our 0-indexed H_in list.
            h_barrier = H_in[k_prob-1]

            if A[idx_from] > A[idx_to] and A[idx_from] > h_barrier:
                A[idx_from] -= 1
                A[idx_to] += 1
                
                # We need to find when A_{k_prob} > 0 for the first time.
                # ans_values[k_prob-1] stores the answer for A_{k_prob}.
                # If A[idx_to] (which is A_{k_prob}) just became 1, it means it was 0.
                if A[idx_to] == 1 and ans_values[k_prob-1] == 0: 
                    ans_values[k_prob-1] = op_count
                    num_ans_found += 1
        
        # Optimization: if all answers found during the inner loop, break early.
        if num_ans_found == N:
            break
            
    print(*(ans_values))

if __name__ == '__main__':
    main()