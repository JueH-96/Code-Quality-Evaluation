import sys

def solve():
    N, W = map(int, sys.stdin.readline().split())
    
    # Store blocks grouped by column and original index
    col_blocks = {x: [] for x in range(1, W + 1)}
    
    for i in range(N):
        X, Y = map(int, sys.stdin.readline().split())
        col_blocks[X].append((Y, i))

    # Sort blocks within each column by Y coordinate
    for x in range(1, W + 1):
        col_blocks[x].sort()

    # Calculate c_x and block_info (original_idx -> (X, Y, k))
    c_vals = {}
    block_info = [None] * N # Store (X, Y, k) for each block index (0-based)
    K_min = float('inf') if W > 0 else 0

    for x in range(1, W + 1):
        c_x = len(col_blocks[x])
        c_vals[x] = c_x
        if W > 0:
             K_min = min(K_min, c_x)

        for k in range(c_x):
            y_val, original_idx_0based = col_blocks[x][k]
            block_info[original_idx_0based] = (x, y_val, k + 1) # k+1 is 1-based rank

    # Calculate T_settle_val(k) = max_{x in [1, W], c_x >= k} (Y_{x,k} - k)
    # This is the maximum "excess height" relative to the k-th row
    # for the k-th block in any column that has at least k blocks.
    # The k-th removal happens when the slowest of these blocks reaches original row k.
    # Time taken = max(Y_x,k - k).
    # T_settle_val_arr[k] stores this max value for rank k.
    # We only need this for k = 1 to K_min, as subsequent removals don't happen.
    
    T_settle_val_arr = [0] * (K_min + 1) # Indices 1 to K_min will be used

    for x in range(1, W + 1):
        for k in range(c_vals[x]):
            rank = k + 1 # 1-based rank
            # We only care about ranks up to K_min because removals stop after K_min levels
            if rank <= K_min:
                y_xk = col_blocks[x][k][0] # Y value of k-th block in column x
                # Y_x,k is the Y coordinate of the block that is the k-th lowest
                # in column x initially. Y_x,k >= k.
                T_settle_val_arr[rank] = max(T_settle_val_arr[rank], y_xk - rank)
                
    # Calculate T_removal(k) = T_settle_val(k) + 1
    # This is the time step AT WHICH the k-th row removal happens.
    # Removal happens at time tau if row 1 is full at time tau - 1.
    # The time taken for the k-th layer of blocks to settle at original row k is T_settle_val(k).
    # Row 1 becomes full for the k-th time after T_settle_val(k) time units have passed
    # since the (k-1)-th removal. The removal then occurs at the next time step.
    T_removal_arr = [0] * (K_min + 1) # Indices 1 to K_min will be used
    for k in range(1, K_min + 1):
        # T_settle_val_arr[k] is the time elapsed *after* the (k-1)-th removal
        # for the k-th blocks to reach row 1 (original row k).
        # The removal happens at the time step that checks the state after this settling.
        # Let t_s = T_settle_val_arr[k] be the duration.
        # If (k-1)-th removal was at time T_{rem}(k-1), then row 1 is full at T_{rem}(k-1) + t_s.
        # Removal happens at T_{rem}(k-1) + t_s + 1? No.
        # The formula derived during thought: T_{removal}(k) = max_{x: c_x >= k} (Y_{x,k} - k) + 1
        T_removal_arr[k] = T_settle_val_arr[k] + 1

    # Process queries
    Q = int(sys.stdin.readline())
    for _ in range(Q):
        T, A = map(int, sys.stdin.readline().split())
        
        # A is 1-based block index, convert to 0-based
        block_idx_0based = A - 1
        
        # Get block info: (X, Y, k)
        X_A, Y_A, k_A = block_info[block_idx_0based]

        # A block with rank k_A is removed during the k_A-th row removal
        # IF the k_A-th row removal actually happens.
        # The k_A-th row removal happens IFF k_A <= K_min.
        
        if k_A > K_min:
            # This block's rank is higher than the minimum column size.
            # The k_A-th row removal never happens globally (because some column has < k_A blocks).
            # This block is never removed by the row clearing rule.
            print("Yes")
        else:
            # This block's rank is k_A <= K_min.
            # It is the k_A-th block in column X_A.
            # The k_A-th row removal event happens at time T_removal_arr[k_A].
            # If the query time T is >= the removal time, the block is gone.
            # The query asks about time T + 0.5.
            # If removal is at time tau, block is removed at exactly time tau.
            # It exists at tau - epsilon, does not exist at tau + epsilon.
            # Query time T + 0.5.
            # If removal time is tau, block exists at T + 0.5 if T + 0.5 < tau.
            # T + 0.5 < tau  <=> T < tau - 0.5
            # Since T and tau are integers, T < tau.
            # This is equivalent to T <= tau - 1.
            # So block exists at T + 0.5 if T < T_removal_arr[k_A] OR T <= T_removal_arr[k_A] - 1.
            # If T_removal_arr[k_A] = 1, removal is at time 1. Block exists at 0.5, not at 1.5.
            # Query T=1, A=1 in Sample 1. T=1, T_removal=2. 1 < 2. Yes.
            # Query T=2, A=3 in Sample 1. T=2, T_removal=2. 2 >= 2. No.
            # The condition T >= T_removal_arr[k_A] for removal is correct.
            if T >= T_removal_arr[k_A]:
                print("No")
            else:
                print("Yes")

solve()