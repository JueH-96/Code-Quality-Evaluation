def solve():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(input().strip()))
    
    MOD = 998244353
    
    # Find all '?' positions
    question_marks = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '?':
                question_marks.append((i, j))
    
    def backtrack(idx):
        if idx == len(question_marks):
            return 1
        
        result = 0
        i, j = question_marks[idx]
        
        for digit in ['1', '2', '3']:
            # Check if this digit conflicts with adjacent cells
            valid = True
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if grid[ni][nj] != '?' and grid[ni][nj] == digit:
                        valid = False
                        break
            
            if valid:
                grid[i][j] = digit
                result = (result + backtrack(idx + 1)) % MOD
                grid[i][j] = '?'  # backtrack
        
        return result
    
    return backtrack(0)

print(solve())