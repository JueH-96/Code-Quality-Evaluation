def solve():
    N, W = map(int, input().split())
    blocks = {}
    for i in range(N):
        x, y = map(int, input().split())
        blocks[i+1] = (x, y)
    
    Q = int(input())
    queries = []
    for _ in range(Q):
        t, a = map(int, input().split())
        queries.append((t, a))
    
    for t, a in queries:
        current_blocks = blocks.copy()
        
        for time_step in range(1, t + 1):
            # Create grid representation
            grid = {}
            for block_id, (x, y) in current_blocks.items():
                if y not in grid:
                    grid[y] = set()
                grid[y].add(x)
            
            # Remove complete bottom rows
            while grid:
                min_y = min(grid.keys())
                if len(grid[min_y]) == W:
                    # Remove blocks in bottom row
                    to_remove = []
                    for block_id, (x, y) in current_blocks.items():
                        if y == min_y:
                            to_remove.append(block_id)
                    for block_id in to_remove:
                        del current_blocks[block_id]
                    del grid[min_y]
                else:
                    break
            
            # Apply gravity
            if current_blocks:
                # Get all positions that are occupied
                occupied_positions = set()
                for x, y in current_blocks.values():
                    occupied_positions.add((x, y))
                
                # Process blocks from bottom to top
                blocks_by_y = {}
                for block_id, (x, y) in current_blocks.items():
                    if y not in blocks_by_y:
                        blocks_by_y[y] = []
                    blocks_by_y[y].append((block_id, x))
                
                new_blocks = {}
                new_occupied = set()
                
                for y in sorted(blocks_by_y.keys()):
                    for block_id, x in blocks_by_y[y]:
                        # Find lowest position this block can fall to
                        target_y = y
                        while target_y > 1 and (x, target_y - 1) not in new_occupied:
                            target_y -= 1
                        
                        new_blocks[block_id] = (x, target_y)
                        new_occupied.add((x, target_y))
                
                current_blocks = new_blocks
        
        if a in current_blocks:
            print("Yes")
        else:
            print("No")

solve()