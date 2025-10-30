def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    W = int(data[idx])
    idx += 1
    
    positions = {}
    for i in range(1, N + 1):
        X = int(data[idx])
        idx += 1
        Y = int(data[idx])
        idx += 1
        if X not in positions:
            positions[X] = {}
        positions[X][Y] = i
    
    Q = int(data[idx])
    idx += 1
    
    queries = []
    for _ in range(Q):
        T = int(data[idx])
        idx += 1
        A = int(data[idx])
        idx += 1
        queries.append((T, A))
    
    # To store the final positions of blocks after all removals
    block_positions = {}
    for x in range(1, W + 1):
        if x in positions:
            # Sort the y-values to process from bottom to top
            ys = sorted(positions[x].keys())
            # Find the lowest y that is continuous from y=1
            lowest_y = 1
            while lowest_y in positions[x]:
                lowest_y += 1
            lowest_y -= 1
            
            # All blocks from y=1 to lowest_y will be removed at time lowest_y
            removal_time = lowest_y
            for y in ys:
                block_id = positions[x][y]
                if y <= lowest_y:
                    # This block will be removed at time `removal_time`
                    block_positions[block_id] = (x, y, removal_time)
                else:
                    # This block will start falling after `removal_time`
                    # It will be at position y - removal_time at time removal_time + 0.5
                    final_y = y - removal_time
                    block_positions[block_id] = (x, final_y, removal_time)
    
    results = []
    for T, A in queries:
        if A in block_positions:
            x, final_y, removal_time = block_positions[A]
            if T >= removal_time:
                results.append("No")
            else:
                results.append("Yes")
        else:
            results.append("No")
    
    print("
".join(results))