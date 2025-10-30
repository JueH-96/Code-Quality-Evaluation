# YOUR CODE HERE
def solve():
    N, W = map(int, input().split())
    
    # Read initial block positions
    blocks = []
    for i in range(N):
        x, y = map(int, input().split())
        blocks.append((x-1, y, i))  # Convert to 0-indexed column, keep 1-indexed row
    
    # Group blocks by column
    columns = [[] for _ in range(W)]
    for x, y, idx in blocks:
        columns[x].append((y, idx))
    
    # Sort each column by row (bottom to top)
    for col in columns:
        col.sort()
    
    # Track when each block is removed (-1 means never removed)
    removal_time = [-1] * N
    
    # Simulate the game
    time = 0
    while True:
        time += 1
        
        # Make blocks fall in each column
        for col_idx in range(W):
            if not columns[col_idx]:
                continue
            
            new_positions = []
            for i, (row, block_idx) in enumerate(columns[col_idx]):
                if i == 0:
                    # First block falls to row 1
                    new_positions.append((1, block_idx))
                else:
                    # Other blocks fall to just above the previous block
                    new_positions.append((new_positions[i-1][0] + 1, block_idx))
            
            columns[col_idx] = new_positions
        
        # Check if bottom row is full
        bottom_row_blocks = []
        for col_idx in range(W):
            if columns[col_idx] and columns[col_idx][0][0] == 1:
                bottom_row_blocks.append((col_idx, columns[col_idx][0][1]))
        
        if len(bottom_row_blocks) == W:
            # Remove all blocks in bottom row
            for col_idx, block_idx in bottom_row_blocks:
                removal_time[block_idx] = time
                columns[col_idx].pop(0)
        
        # Check if all blocks are removed or if we've simulated enough
        all_removed = all(len(col) == 0 for col in columns)
        if all_removed or time > 10**9:
            break
    
    # Process queries
    Q = int(input())
    for _ in range(Q):
        T, A = map(int, input().split())
        A -= 1  # Convert to 0-indexed
        
        if removal_time[A] == -1 or removal_time[A] > T:
            print("Yes")
        else:
            print("No")

solve()