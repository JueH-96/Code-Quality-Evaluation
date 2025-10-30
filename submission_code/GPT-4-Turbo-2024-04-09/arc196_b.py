MOD = 998244353

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        H = int(data[index])
        W = int(data[index + 1])
        index += 2
        
        grid = []
        for _ in range(H):
            grid.append(data[index])
            index += 1
        
        # Check if there's a valid tiling configuration
        # We need to check if the grid can be tiled with no dead ends considering toroidal connections
        
        # We can use a union-find or a direct check approach
        # Since the grid is small, we can try to directly determine the configuration
        
        # Check for horizontal and vertical matching
        horizontal_match = True
        vertical_match = True
        
        # Check horizontal wraps
        for i in range(H):
            if grid[i][0] != grid[i][-1]:
                horizontal_match = False
                break
        
        # Check vertical wraps
        for j in range(W):
            if grid[0][j] != grid[-1][j]:
                vertical_match = False
                break
        
        if not horizontal_match or not vertical_match:
            results.append(0)
            continue
        
        # If we pass the wrap checks, we need to count the number of A's and B's
        count_A = 0
        count_B = 0
        for i in range(H):
            for j in range(W):
                if grid[i][j] == 'A':
                    count_A += 1
                else:
                    count_B += 1
        
        # Calculate the number of ways to place the tiles
        # 4^count_A * 2^count_B % MOD
        result = pow(4, count_A, MOD) * pow(2, count_B, MOD) % MOD
        results.append(result)
    
    # Output all results
    sys.stdout.write('
'.join(map(str, results)) + '
')