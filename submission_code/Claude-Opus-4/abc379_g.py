def solve():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(input().strip()))
    
    MOD = 998244353
    
    # Find all '?' positions
    questions = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '?':
                questions.append((i, j))
    
    def is_valid(r, c, val):
        # Check if placing val at (r, c) is valid
        # Check up
        if r > 0 and grid[r-1][c] != '?' and grid[r-1][c] == val:
            return False
        # Check down
        if r < H-1 and grid[r+1][c] != '?' and grid[r+1][c] == val:
            return False
        # Check left
        if c > 0 and grid[r][c-1] != '?' and grid[r][c-1] == val:
            return False
        # Check right
        if c < W-1 and grid[r][c+1] != '?' and grid[r][c+1] == val:
            return False
        return True
    
    def backtrack(idx):
        if idx == len(questions):
            # Check if the complete grid is valid
            for i in range(H):
                for j in range(W):
                    val = grid[i][j]
                    # Check right neighbor
                    if j < W-1 and grid[i][j+1] == val:
                        return 0
                    # Check down neighbor
                    if i < H-1 and grid[i+1][j] == val:
                        return 0
            return 1
        
        r, c = questions[idx]
        count = 0
        
        for val in ['1', '2', '3']:
            if is_valid(r, c, val):
                grid[r][c] = val
                count = (count + backtrack(idx + 1)) % MOD
                grid[r][c] = '?'
        
        return count
    
    # First check if the initial grid (without considering ?) is valid
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '?':
                val = grid[i][j]
                # Check right neighbor
                if j < W-1 and grid[i][j+1] != '?' and grid[i][j+1] == val:
                    print(0)
                    return
                # Check down neighbor
                if i < H-1 and grid[i+1][j] != '?' and grid[i+1][j] == val:
                    print(0)
                    return
    
    result = backtrack(0)
    print(result)

solve()