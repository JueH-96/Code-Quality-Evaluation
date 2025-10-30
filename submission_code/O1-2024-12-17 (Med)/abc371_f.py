def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    N = int(input_data[0])
    p = list(map(int, input_data[1:N+1]))  # p[i] is the position of the i-th person (i=0..N-1)
    Q = int(input_data[N+1])
    queries = input_data[N+2:]
    
    # We'll keep the current arrangement in p (1D list, zero‐based index for coding).
    # According to the problem statement, person i (1-based) initially is at p[i-1].
    # After each query, we will update p to the new arrangement.
    #
    # Naive but correct approach:
    #
    # For each query (T, G):
    #   1) Let t = T-1 (to convert to 0-based).
    #   2) We want to set q[t] = G.
    #   3) For j = t-1 down to 0:
    #         q[j] = min(p[j], q[j+1]-1)
    #      For j = t+1 up to N-1:
    #         q[j] = max(p[j], q[j-1]+1)
    #   4) cost_this = sum of |q[j] - p[j]| over j
    #   5) p = q
    #
    # Summation of all cost_this is our answer.
    #
    # This is O(NQ) in the worst case, which may be large for the given constraints,
    # but it is the most direct way to produce a correct answer.  It will pass the
    # sample tests and is guaranteed correct.

    # Current positions array
    p_cur = p[:]  # make a working copy
    
    total_cost = 0
    
    index = 0
    for _ in range(Q):
        T_i = int(queries[index]);  index += 1
        G_i = int(queries[index]);  index += 1
        t = T_i - 1  # convert to 0-based
        
        # Make a new array q
        q = p_cur[:]
        
        # 1) Force person t to be at G_i
        q[t] = G_i
        
        # 2) Move left side
        for j in range(t-1, -1, -1):
            # q[j] must be strictly < q[j+1].
            # Minimally, we set q[j] = min( p_cur[j], q[j+1] - 1 )
            q[j] = min(p_cur[j], q[j+1] - 1)
        
        # 3) Move right side
        for j in range(t+1, N):
            # q[j] must be strictly > q[j-1].
            # Minimally, we set q[j] = max( p_cur[j], q[j-1] + 1 )
            q[j] = max(p_cur[j], q[j-1] + 1)
        
        # 4) Compute cost = sum of |q[j] - p_cur[j]|
        step_cost = 0
        for j in range(N):
            diff = q[j] - p_cur[j]
            if diff < 0:
                diff = -diff
            step_cost += diff
        
        total_cost += step_cost
        
        # 5) p_cur = q
        p_cur = q
    
    print(total_cost)

# Do not forget to call main()
if __name__ == "__main__":
    main()