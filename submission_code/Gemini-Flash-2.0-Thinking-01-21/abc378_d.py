def solve():
    h, w, k = map(int, input().split())
    grid = [input() for _ in range(h)]
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w and grid[r][c] == '.'
        
    def count_paths_from(start_r, start_c):
        count = 0
        initial_visited = set([(start_r, start_c)])
        
        def dfs(current_r, current_c, moves_made, visited_cells):
            nonlocal count
            if moves_made == k:
                return 1
            
            path_count = 0
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_r, next_c = current_r + dr, current_c + dc
                if is_valid(next_r, next_c) and (next_r, next_c) not in visited_cells:
                    next_visited_cells = set(visited_cells)
                    next_visited_cells.add((next_r, next_c))
                    path_count += dfs(next_r, next_c, moves_made + 1, next_visited_cells)
            return path_count
            
        return dfs(start_r, start_c, 0, initial_visited)
        
    total_paths = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == '.':
                total_paths += count_paths_from(r, c)
                
    print(total_paths)

if __name__ == '__main__':
    solve()