import sys

def solve():
    """
    Solves the Good Sequence problem by breaking it down into independent segments
    and using dynamic programming on each segment.
    """
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # d_i is the required shift for A_i to become B_i, modulo M.
    # d_i = (B_i - A_i) mod M
    d = [(B[i] - A[i] + M) % M for i in range(N)]

    if N == 1:
        print(min(d[0], M - d[0]))
        return

    # delta_A_i = (A_i - A_{i-1}) mod M
    delta_A = [(A[i] - A[i-1] + M) % M for i in range(1, N)]
    delta_B = [(B[i] - B[i-1] + M) % M for i in range(1, N)]
    
    # Find separators where the problem can be split.
    # This happens when there's no constraint on k_i - k_{i-1}.
    # which is when delta_A[i-1] == delta_B[i-1].
    # The segments are between these separators.
    separators = [-1]
    for i in range(N - 1):
        if delta_A[i] == delta_B[i]:
            separators.append(i)
    separators.append(N - 1)

    total_cost = 0
    for seg_idx in range(len(separators) - 1):
        start_node = separators[seg_idx] + 1
        end_node = separators[seg_idx + 1]

        if start_node > end_node:
            continue
        
        # DP on the segment [start_node, end_node]
        # DP state: m_prev = optimal k_{i-1}, cost_prev = min cost for prefix
        
        # Base case for DP at start_node
        d_start = d[start_node]
        # The cost |d_start + k*M| is minimized at k=0 or k=-1
        if 2 * d_start <= M:
            m_prev = 0
        else:
            m_prev = -1
        cost_prev = abs(d_start + m_prev * M)

        for i in range(start_node + 1, end_node + 1):
            da = delta_A[i-1]
            db = delta_B[i-1]
            
            d_curr = d[i]
            d_prev = d[i-1]
            diff_d = d_curr - d_prev

            l_i, u_i = -float('inf'), float('inf')

            if da < db: # Constraint: s_i - s_{i-1} > 0
                # k_i - k_{i-1} >= floor((-diff_d)/M) + 1
                l_i = (-diff_d) // M + 1
            elif da > db: # Constraint: s_i - s_{i-1} < 0
                # k_i - k_{i-1} <= floor((-diff_d-1)/M)
                u_i = (-diff_d - 1) // M
            
            # Find locally optimal k for current node i
            if 2 * d_curr <= M:
                k_min_curr = 0
            else:
                k_min_curr = -1

            # Find optimal m_curr by clipping the local optimum to the valid interval
            # derived from the constraint with the previous optimal k, m_prev.
            min_I = m_prev + l_i
            max_I = m_prev + u_i
            
            m_curr = k_min_curr
            if m_curr < min_I: m_curr = min_I
            if m_curr > max_I: m_curr = max_I

            # The total cost for the prefix up to i is the cost up to i-1
            # plus the cost for element i, plus any additional cost incurred
            # from the interaction between i-1 and i.
            # Let DP(i, k) be the min cost for prefix ending at i with k_i = k.
            # DP(i, k) = |d_i + kM| + min_{k' in range} DP(i-1, k').
            # The minimum of DP(i-1, k') is cost_{i-1} at k'=m_{i-1}.
            # The additional cost is from moving the minimum from m_{i-1} to a
            # k' that satisfies the constraint with m_curr.
            dist = 0
            # The valid range for k_{i-1} is [m_curr - u_i, m_curr - l_i]
            if m_prev < m_curr - u_i:
                dist = (m_curr - u_i) - m_prev
            if m_prev > m_curr - l_i:
                dist = m_prev - (m_curr - l_i)
            
            cost_curr = cost_prev + abs(d_curr + m_curr * M) + M * dist
            
            cost_prev = cost_curr
            m_prev = m_curr

        total_cost += cost_prev

    print(total_cost)

solve()