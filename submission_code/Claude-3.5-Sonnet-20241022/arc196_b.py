def solve_case():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    MOD = 998244353
    
    # For type A tiles: 0=┌, 1=┐, 2=└, 3=┘
    # For type B tiles: 0=horizontal, 1=vertical
    
    def check_valid(state):
        # Check if current state is valid (no dead ends)
        for i in range(H):
            for j in range(W):
                # Get current tile type and orientation
                tile = grid[i][j]
                orient = state[i][j]
                
                # Check right connection
                has_right = False
                if tile == 'A':
                    has_right = orient in [1, 2]  # ┐ or └
                else:  # tile == 'B'
                    has_right = orient == 0  # horizontal
                
                next_j = (j + 1) % W
                next_tile = grid[i][next_j]
                next_orient = state[i][next_j]
                
                has_left = False
                if next_tile == 'A':
                    has_left = next_orient in [0, 3]  # ┌ or ┘
                else:  # next_tile == 'B'
                    has_left = next_orient == 0  # horizontal
                
                if has_right != has_left:
                    return False
                
                # Check bottom connection
                has_bottom = False
                if tile == 'A':
                    has_bottom = orient in [2, 3]  # └ or ┘
                else:  # tile == 'B'
                    has_bottom = orient == 1  # vertical
                
                next_i = (i + 1) % H
                next_tile = grid[next_i][j]
                next_orient = state[next_i][j]
                
                has_top = False
                if next_tile == 'A':
                    has_top = next_orient in [0, 1]  # ┌ or ┐
                else:  # next_tile == 'B'
                    has_top = next_orient == 1  # vertical
                
                if has_bottom != has_top:
                    return False
        
        return True

    def dfs(i, j, state):
        if i == H:
            return 1 if check_valid(state) else 0
            
        next_i = i + (j + 1) // W
        next_j = (j + 1) % W
        
        result = 0
        if grid[i][j] == 'A':
            # Try all 4 orientations for type A
            for orient in range(4):
                state[i][j] = orient
                result = (result + dfs(next_i, next_j, state)) % MOD
        else:  # grid[i][j] == 'B'
            # Try both orientations for type B
            for orient in range(2):
                state[i][j] = orient
                result = (result + dfs(next_i, next_j, state)) % MOD
        
        return result
    
    # Initialize state array
    state = [[0] * W for _ in range(H)]
    answer = dfs(0, 0, state)
    print(answer)

# Read number of test cases
T = int(input())
for _ in range(T):
    solve_case()