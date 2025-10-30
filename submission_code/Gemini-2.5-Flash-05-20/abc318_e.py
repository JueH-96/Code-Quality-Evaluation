import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Step 1: Group indices by value
    # positions[value] will be a sorted list of 0-based indices where A[idx] == value
    positions = defaultdict(list)
    for i in range(N):
        positions[A[i]].append(i)

    total_triples = 0

    # Step 2: Iterate through each unique value
    for value, idx_list in positions.items():
        m = len(idx_list) # m is the number of occurrences of 'value'

        # We need at least two occurrences to form an (i, k) pair (A_i = A_k).
        # If m < 2, no such pairs are possible for this value.
        if m < 2:
            continue

        # Calculate the first part of the sum: S_v_pos_idx = sum_{k=0}^{m-1} (2k - (m-1)) * p_k
        # This part accounts for the values of A_i and A_k (which are p_x and p_y here).
        current_value_contribution = 0
        for k in range(m):
            p_k = idx_list[k]
            current_value_contribution += (2 * k - (m - 1)) * p_k
        
        # Calculate the second part of the sum: S_v_neg_idx = -(m-1)m(m+1)//6
        # This part accounts for the 'y - x' component from the formula k - i - (y - x).
        # It's derived from summing -(y-x) over all pairs (x,y) for the current value.
        # This simplifies to -1/2 * sum_{y=1}^{m-1} y(y+1), which is -1/2 * (m-1)m(m+1)/3.
        # Thus, it's -(m-1)m(m+1)/6.
        numeric_part = -((m - 1) * m * (m + 1) // 6)

        total_triples += current_value_contribution + numeric_part

    print(total_triples)

solve()