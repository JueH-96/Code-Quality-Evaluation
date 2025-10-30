# YOUR CODE HERE
def solve():
    N, W = map(int, input().split())
    
    # Store blocks as (X, Y, id)
    blocks = []
    for i in range(N):
        X, Y = map(int, input().split())
        blocks.append((X, Y, i + 1))
    
    # Process queries
    Q = int(input())
    queries = []
    for j in range(Q):
        T, A = map(int, input().split())
        queries.append((T, A))
    
    # Initialize block positions and track which ones are still alive
    block_pos = {}
    for X, Y, block_id in blocks:
        block_pos[block_id] = (X, Y)
    
    alive = set(range(1, N + 1))
    
    # Group queries by time to process them efficiently
    queries_by_time = {}
    for T, A in queries:
        if T not in queries_by_time:
            queries_by_time[T] = []
        queries_by_time[T].append(A)
    
    # Find the maximum time we need to simulate
    max_time = max(T for T, _ in queries)
    
    # Simulate the grid for each time step up to max_time
    time = 0
    while time <= max_time:
        # Check if the bottom row is filled
        bottom_filled = True
        for X in range(1, W + 1):
            col_has_block = False
            for block_id in alive:
                if block_pos[block_id][0] == X and block_pos[block_id][1] == 1:
                    col_has_block = True
                    break
            if not col_has_block:
                bottom_filled = False
                break
        
        if bottom_filled:
            # Remove all blocks in the bottom row
            to_remove = []
            for block_id in alive:
                if block_pos[block_id][1] == 1:
                    to_remove.append(block_id)
            for block_id in to_remove:
                alive.remove(block_id)
        
        # Answer queries for the current time
        if time in queries_by_time:
            for A in queries_by_time[time]:
                print("Yes" if A in alive else "No")
        
        if time == max_time:
            break
        
        # Calculate block movements
        block_movements = []
        for block_id in alive:
            X, Y = block_pos[block_id]
            if Y == 1:
                continue  # Already at the bottom
            
            # Check if there's a block directly below
            block_below = False
            for other_id in alive:
                if block_pos[other_id][0] == X and block_pos[other_id][1] == Y - 1:
                    block_below = True
                    break
            
            if not block_below:
                # Move the block one cell downward
                block_movements.append((block_id, (X, Y - 1)))
        
        # Apply all movements
        for block_id, new_pos in block_movements:
            block_pos[block_id] = new_pos
        
        time += 1

solve()