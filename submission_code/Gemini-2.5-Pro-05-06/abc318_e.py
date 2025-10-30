import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    value_to_indices = defaultdict(list)
    for i in range(N):
        value_to_indices[A[i]].append(i) # Store 0-based indices

    total_triples = 0

    for value_x in value_to_indices:
        pos_x = value_to_indices[value_x] # List of 0-based indices where A[idx] == value_x
        m = len(pos_x) # Number of times value_x appears in A

        if m < 2:
            # Need at least two occurrences of value_x to form (i, k)
            continue

        # Sum to calculate: Sum_{t=0}^{m-1} (2*t - m + 1) * v_t
        # where v_t = pos_x[t] - t.
        # 't' here is the loop variable k_idx.
        
        current_sum_for_this_value_x = 0
        for k_idx in range(m):
            # p_actual_idx is pos_x[k_idx], the actual 0-based index in original array A.
            p_actual_idx = pos_x[k_idx] 
            
            # v_k = (A's index of k_idx-th occurrence of value_x) - (k_idx itself)
            v_k = p_actual_idx - k_idx
            
            coefficient = (2 * k_idx - m + 1)
            current_sum_for_this_value_x += coefficient * v_k
        
        total_triples += current_sum_for_this_value_x
        
    print(total_triples)

if __name__ == '__main__':
    solve()