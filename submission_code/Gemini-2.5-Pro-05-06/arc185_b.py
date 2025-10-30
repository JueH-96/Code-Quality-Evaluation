import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    current_sum_A_prefix = 0
    S = 0
    # Calculate S. Using a loop to avoid creating a large temporary sum object for potentially large N.
    # For competitive Python, sum(A) is usually fine.
    for x in A:
        S += x

    q = S // N
    r = S % N  # S = N*q + r

    # The target non-decreasing sequence B (B_fair) consists of:
    # N-r elements equal to q
    # r elements equal to q+1
    # Sorted: B_fair = [q, q, ..., q (N-r times), q+1, ..., q+1 (r times)]
    # N_minus_r is the count of 'q' values at the beginning of B_fair
    N_minus_r = N - r

    possible = True
    for k_idx in range(N):  # k_idx from 0 to N-1
        k = k_idx + 1  # k from 1 to N
        
        current_sum_A_prefix += A[k_idx]  # This is P_k, the k-th prefix sum of A

        # Calculate Q_k for B_fair using formula: Q_k = k*q + max(0, k - (N-r))
        q_k_fair = k * q + max(0, k - N_minus_r)
        
        if k == N:
            # For k=N, P_N = S and Q_N = S.
            # The condition P_N <= Q_N becomes S <= S, which is always true.
            # No check needed here for P_N <= Q_N.
            # Also, the formula for q_k_fair when k=N:
            # q_N_fair = N*q + max(0, N - (N-r)) = N*q + max(0, r)
            # Since r >= 0 by definition of modulo, this is N*q + r = S. Correct.
            pass
        else:
            # For k < N, check P_k <= Q_k
            if current_sum_A_prefix > q_k_fair:
                possible = False
                break
    
    if possible:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

num_test_cases = int(sys.stdin.readline())
for _ in range(num_test_cases):
    solve()