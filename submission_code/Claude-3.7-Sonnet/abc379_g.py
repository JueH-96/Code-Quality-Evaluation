def main():
    H, W = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(list(input().strip()))
    
    MOD = 998244353
    
    # Find all cells with question marks
    question_marks = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '?':
                question_marks.append((i, j))
    
    # Check if the initial grid is valid
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '?':
                for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '?' and grid[ni][nj] == grid[i][j]:
                        print(0)
                        return
    
    # Use memoization to avoid recalculating states
    memo = {}
    
    def dfs(idx):
        if idx == len(question_marks):
            return 1
        
        if idx in memo:
            return memo[idx]
        
        i, j = question_marks[idx]
        count = 0
        for d in range(1, 4):
            digit = str(d)
            valid = True
            for ni, nj in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '?' and grid[ni][nj] == digit:
                    valid = False
                    break
            
            if valid:
                grid[i][j] = digit
                count = (count + dfs(idx + 1)) % MOD
                grid[i][j] = '?'  # Backtrack
        
        memo[idx] = count
        return count
    
    print(dfs(0))

if __name__ == "__main__":
    main()