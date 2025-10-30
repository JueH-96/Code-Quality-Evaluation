def solve():
    H, W, K = map(int, input().split())
    grid = [input() for _ in range(H)]
    
    def count_paths(start_row, start_col):
        count = 0
        
        def is_valid(row, col):
            return 0 <= row < H and 0 <= col < W and grid[row][col] == '.'
        
        def backtrack(row, col, path):
            nonlocal count
            
            if len(path) == K + 1:
                count += 1
                return
            
            # Possible moves: up, down, left, right
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for dr, dc in moves:
                new_row = row + dr
                new_col = col + dc
                
                if is_valid(new_row, new_col) and (new_row, new_col) not in path:
                    backtrack(new_row, new_col, path + [(new_row, new_col)])
        
        backtrack(start_row, start_col, [(start_row, start_col)])
        return count
    
    total_paths = 0
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '.':
                total_paths += count_paths(r, c)
    
    print(total_paths)

solve()