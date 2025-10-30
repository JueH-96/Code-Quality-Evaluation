import sys

def solve():
    N, S = map(int, sys.stdin.readline().split())
    # Use 0-indexed A internally
    A_list = list(map(int, sys.stdin.readline().split())) 

    sum_N = sum(A_list)
    # Since A_i >= 1, sum_N >= N >= 1. This means sum_N cannot be 0.

    # S_base is the smallest positive integer S_0 such that S_0 === S (mod sum_N).
    # S_0 represents a sum Q_k - Q_j, which must be positive.
    if S % sum_N == 0:
        # S is a positive multiple of sum_N (since S >= 1, sum_N >= 1)
        S_base = sum_N 
    else:
        # S is not a multiple of sum_N
        S_base = S % sum_N
        
    # seen_prefix_sums stores Q_j values encountered so far (for j < k).
    # Q_j are actual prefix sums of B = (A_0...A_{N-1}, A_0...A_{N-1}), not modulo sum_N.
    # current_Q will hold Q_k = sum of B_0...B_{k-1}.
    
    seen_prefix_sums = {0} # Stores Q_0 initially
    current_Q = 0          # This is Q_0

    # Loop generates Q_1, Q_2, ..., Q_{2N} by processing elements B_0, B_1, ..., B_{2N-1}.
    # In iteration k_loop_idx (0 to 2N-1), current_Q becomes Q_{k_loop_idx + 1} using B_{k_loop_idx}.
    for k_loop_idx in range(2 * N): 
        
        val_B = A_list[k_loop_idx % N] # This is B[k_loop_idx]
        current_Q += val_B             # current_Q is now Q_{k_loop_idx + 1}

        # We are looking for a Q_j (from seen_prefix_sums) such that current_Q - Q_j = S_0.
        # S_0 must satisfy: S_0 === S (mod sum_N), S_0 > 0, S_0 <= S.
        # The candidates for S_0 are S_base and S_base + sum_N, provided they are <= current_Q.
        # (max S_0 is Q_{2N} - Q_0 = 2 * sum_N).

        # Candidate S_0 #1: S_base
        # It's proven that S_base <= S for S >= 1.
        if S_base <= current_Q: 
            target_Q_j = current_Q - S_base
            if target_Q_j in seen_prefix_sums:
                print("Yes")
                return

        # Candidate S_0 #2: S_base + sum_N
        s0_candidate2 = S_base + sum_N
        # This candidate must also satisfy S_0 <= S.
        if s0_candidate2 <= S and s0_candidate2 <= current_Q:
            target_Q_j = current_Q - s0_candidate2
            if target_Q_j in seen_prefix_sums:
                print("Yes")
                return
        
        seen_prefix_sums.add(current_Q) # Store Q_{k_loop_idx + 1} for future iterations

    print("No")

solve()