import sys

# Read input functions for potentially faster I/O
# input = sys.stdin.readline 
# print = sys.stdout.write

def solve():
    """
    Solves the block falling and removal problem based on the given specification.
    Calculates the removal time for blocks that get removed and answers queries
    about block existence at specified times.
    """
    
    N, W = map(int, sys.stdin.readline().split())
    
    # Dictionary to store blocks grouped by column X. 
    # Key: column index X, Value: list of blocks {'Y': Y, 'id': original_idx}
    blocks_in_col = {} 
    
    # List to store final processed info for each block, indexed by original_idx (0 to N-1)
    initial_blocks_data = [None] * N 

    # Read block coordinates and populate blocks_in_col dictionary
    for i in range(N):
        X, Y = map(int, sys.stdin.readline().split())
        
        # Store basic block info; rank and V value will be added later
        initial_blocks_data[i] = {'X': X, 'Y': Y, 'id': i} 
        
        # Add block to the list for its column X
        if X not in blocks_in_col:
            blocks_in_col[X] = []
        blocks_in_col[X].append({'Y': Y, 'id': i})

    # Dictionary to store the rank (1-based vertical position) for each block by its original_idx
    block_ranks = {} 
    
    # Dictionary to store the height (number of blocks) for each column X
    col_heights = {} 
    
    # Dictionary to store the calculated V value for each block. Key is tuple (X, rank), Value is Y_{x,p} - p
    V_values = {} 

    # Initialize the minimum height found across all columns. If a column has 0 blocks, min_H will be 0.
    min_H = N # Maximum possible blocks in a column is N (if W=1)

    # Get list of columns that actually contain blocks
    relevant_cols = sorted(blocks_in_col.keys())

    # Process each column with blocks to determine ranks and calculate V values
    for x in relevant_cols:
        blocks = blocks_in_col[x]
        
        # Sort blocks within the column by Y coordinate to determine their rank (1-based index from bottom)
        blocks.sort(key=lambda b: b['Y'])
        
        H_x = len(blocks)
        col_heights[x] = H_x
        min_H = min(min_H, H_x) # Update the overall minimum column height
        
        # Assign ranks and calculate V values for each block in this column
        for p in range(H_x):
            rank = p + 1 # 1-based rank
            block_idx = blocks[p]['id'] # Original index (0-based)
            block_Y = blocks[p]['Y'] # Initial Y coordinate
            
            # Store the rank for this block
            block_ranks[block_idx] = rank
            initial_blocks_data[block_idx]['p'] = rank # Update main block data structure

            # Calculate V = Y - rank
            V_val = block_Y - rank
            V_values[(x, rank)] = V_val
            initial_blocks_data[block_idx]['V'] = V_val # Store V value in main block data


    # If the number of columns with blocks is less than W, it means some columns are empty.
    # In this case, the minimum height across *all* W columns is 0.
    if len(relevant_cols) < W:
         min_H = 0
    
    # R is the total number of full rows that will eventually be removed.
    # This is equal to the minimum height across all W columns.
    R = min_H

    # If R=0, no removals ever occur. All blocks exist indefinitely.
    if R == 0:
        Q = int(sys.stdin.readline())
        for _ in range(Q):
            # Read query details (T, A). T is time, A is 1-based block index.
            T, A = map(int, sys.stdin.readline().split()) 
            # Since no blocks are ever removed, the answer is always Yes.
            sys.stdout.write("Yes
")
        return # Exit after processing queries for R=0 case

    # If R > 0, calculate removal times T_remove[k] for k=1..R
    
    # Step 1: Compute P_j = max_{x s.t. H_x >= j} (Y_{x,j} - j) for j=1..R
    # P[j] represents the maximum V value among all blocks with rank j.
    P = [-float('inf')] * (R + 1) # P[0] is unused
    # Iterate over columns that have blocks
    for x in relevant_cols:
        max_rank_in_col = col_heights[x]
        # Column x contributes V values for ranks up to min(R, H_x)
        # We only need to consider ranks k from 1 up to R.
        # And for a given column x, only up to its height H_x.
        for k in range(1, min(R, max_rank_in_col) + 1):
             # Update P[k] with the maximum V value found for rank k.
             P[k] = max(P[k], V_values[(x, k)])

    # Step 2: Compute M_k = max_{1<=j<=k} P_j for k=1..R
    # M[k] is the maximum P[j] value for j from 1 to k. This involves prefix maximums.
    M = [-float('inf')] * (R + 1) # M[0] is unused
    current_M_max = -float('inf') # Use -infinity for max over empty set or negative numbers
    for k in range(1, R + 1):
         # Update the running maximum value seen so far
         current_M_max = max(current_M_max, P[k])
         M[k] = current_M_max

    # Step 3: Compute removal times T_remove[k] using the derived formula:
    # T_remove[k] = k + max(0, max_{1<=j<=k} (M[j] + 1 - j))
    T_remove = [0] * (R + 1) # T_remove[0] is conceptually 0. T_remove[k] stores time for k-th removal.
    
    # Keep track of the maximum value of (M_j + 1 - j) seen in the prefix 1..k
    current_Z_max = -float('inf') 
    for k in range(1, R + 1):
        # Calculate the term (M_k + 1 - k) for the current k
        val_Mj_1_minus_j = M[k] + 1 - k
        
        # Update the running maximum of these terms
        current_Z_max = max(current_Z_max, val_Mj_1_minus_j)
        
        # Zk = max(0, current prefix maximum)
        Zk = max(0, current_Z_max)
        
        # Calculate the removal time for the k-th row
        T_remove[k] = k + Zk

    # Store the calculated removal time for each block that gets removed (rank <= R)
    removal_times = {} # Map original_idx -> T_remove time
    for i in range(N):
        # Check if block i has a rank (it should if processed correctly)
        rank = block_ranks.get(i)
        # If the block's rank is less than or equal to R, it will be removed
        if rank is not None and rank <= R:
            # Store its removal time, which corresponds to the removal time of the row matching its rank
            removal_times[i] = T_remove[rank]

    # Process the Q queries
    Q = int(sys.stdin.readline())
    results = [] # Store results to print at the end potentially
    for _ in range(Q):
        T, A = map(int, sys.stdin.readline().split())
        block_idx = A - 1 # Convert 1-based block index A to 0-based index
        
        rank = block_ranks.get(block_idx) # Retrieve the rank of the queried block
        
        # Defensive check: If somehow rank is None, handle it. Under constraints, this shouldn't happen.
        if rank is None:
           results.append("No") # Default guess if data is inconsistent
           continue

        # If block's rank is greater than R, it survives indefinitely
        if rank > R: 
            results.append("Yes")
        else: # Block has rank <= R, so it will be removed
            # Get the removal time for this block
            RT = removal_times[block_idx]
            # Check if the query time T is strictly before the removal time RT
            if T < RT: 
                # Block exists at time T and will be removed at time RT >= T+1. 
                # So it exists at time T + 0.5.
                results.append("Yes")
            else: # Query time T is at or after the removal time RT (T >= RT)
                # Block was removed at time RT <= T. It does not exist at T + 0.5.
                results.append("No")

    # Print all query results
    for res in results:
        sys.stdout.write(res + "
")

# Execute the solver function
solve()