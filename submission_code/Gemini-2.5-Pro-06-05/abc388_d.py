# YOUR CODE HERE
import sys

def solve():
    """
    Solves the alien stones problem by efficiently calculating gift exchanges
    and determining the final stone counts.
    """
    try:
        # Fast I/O
        input = sys.stdin.readline
        N_str = input()
        if not N_str: return
        N = int(N_str)
        if N == 0:
            return
        A = list(map(int, input().split()))
    except (IOError, ValueError):
        # Handles cases with empty input or parsing errors
        return

    # The "giving potential" V_i = A_i + S_i + i can be at most:
    # 5e5 (for A_i) + 5e5 (for S_i < N) + 5e5 (for i < N) = 1.5e6
    # We set a safe upper bound for the Fenwick tree's size.
    V_MAX = 1_500_001
    tree = [0] * (V_MAX + 1)

    # Fenwick Tree (BIT) helper functions
    def add(idx, val):
        """Adds val to index idx in the BIT."""
        idx += 1  # 1-based indexing for the tree
        while idx <= V_MAX:
            tree[idx] += val
            idx += idx & -idx

    def query(idx):
        """Queries the cumulative sum up to index idx."""
        if idx < 0:
            return 0
        idx += 1  # 1-based indexing for the tree
        s = 0
        while idx > 0:
            s += tree[idx]
            idx -= idx & -idx
        return s

    # C[i] will store stones alien i has right after becoming an adult.
    C = [0] * N
    # total_added is the count of V_j values added to the BIT.
    total_added = 0
    
    # Pass 1: Calculate S_i (gifts received) and C_i for each alien.
    for i in range(N):
        # S_i = number of givers = count({j < i | V_j > i - 1})
        # This is total_added - count({j < i | V_j <= i - 1}).
        threshold = i - 1
        count_le_thresh = query(threshold)
        s_i = total_added - count_le_thresh
        
        C[i] = A[i] + s_i
        
        # v_i = C_i + i is the "giving potential".
        v_i = C[i] + i
        
        # Add v_i to BIT. If v_i exceeds V_MAX, cap it, as it will
        # never be passed by the threshold `i` anyway.
        if v_i >= V_MAX:
            v_i = V_MAX - 1
        add(v_i, 1)
        total_added += 1

    # Pass 2: Calculate final stones B_i after N years.
    B = [0] * N
    for i in range(N):
        # Alien i is asked to give gifts for ceremonies from year i+2 to N.
        # Number of requests = (N - (i+2) + 1) = N - 1 - i.
        requests = N - 1 - i
        # Final stones = stones at adulthood - gifts given.
        # Gifts given = min(stones at adulthood, requests).
        # B[i] = C[i] - min(C[i], requests) = max(0, C[i] - requests).
        B[i] = max(0, C[i] - requests)
        
    print(*B)

solve()