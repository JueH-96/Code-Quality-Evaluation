H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(input().strip()))

MOD = 998244353

# Check if the initial grid is already invalid
def is_initially_valid():
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '?':
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] != '?' and grid[ni][nj] == grid[i][j]:
                            return False
    return True

if not is_initially_valid():
    print(0)
else:
    # Find all '?' positions
    questions = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '?':
                questions.append((i, j))
    
    def get_valid_values(i, j):
        valid = {'1', '2', '3'}
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '?':
                valid.discard(grid[ni][nj])
        return list(valid)
    
    def solve(idx):
        if idx == len(questions):
            return 1
        
        i, j = questions[idx]
        valid_values = get_valid_values(i, j)
        
        if not valid_values:
            return 0
        
        count = 0
        for val in valid_values:
            grid[i][j] = val
            count = (count + solve(idx + 1)) % MOD
            grid[i][j] = '?'
        
        return count
    
    print(solve(0))