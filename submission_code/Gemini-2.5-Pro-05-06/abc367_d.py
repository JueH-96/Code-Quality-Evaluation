import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A_input = list(map(int, sys.stdin.readline().split()))

    # R_values[k] stores (A_input[0] + ... + A_input[k-1]) % M
    # R_values[0] is for the empty sum (position of node 1), which is 0.
    R_values = [0] * N 
    current_sum = 0 # Represents sum A_input[0] + ... + A_input[k-1]

    # Calculate R_values elements
    # R_values[0] is already 0, representing P_0 % M.
    for k in range(N - 1): # k from 0 to N-2
        current_sum = current_sum + A_input[k]
        R_values[k+1] = current_sum % M
    
    # Total circumference C
    # current_sum after loop is A_input[0] + ... + A_input[N-2]
    # (This requires N-1 >= 0, so N >= 1. Constraint is N >= 2)
    total_circumference_sum = current_sum + A_input[N-1]
    C_M = total_circumference_sum % M
    
    ans = 0

    # Case 1: s < t
    # This means indices i < j for P_values. R_values[j] == R_values[i].
    # Iterate k as j (current index), count previous i's.
    counts1 = [0] * M
    for k in range(N): # k iterates through indices of R_values (0 to N-1)
        val_R_k = R_values[k]
        ans += counts1[val_R_k]
        counts1[val_R_k] += 1
        
    # Case 2: s > t
    # This means indices j < i for P_values. R_values[j] == (R_values[i] - C_M + M) % M.
    # Iterate k as i (current index), count previous j's.
    counts2 = [0] * M
    for k in range(N): # k iterates through indices of R_values (0 to N-1)
        val_R_k = R_values[k] 
        target_R_j = (val_R_k - C_M + M) % M
        ans += counts2[target_R_j]
        counts2[val_R_k] += 1

    print(ans)

solve()