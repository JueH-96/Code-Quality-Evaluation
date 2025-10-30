def solve():
    h, w, k = map(int, input().split())
    grid = [input() for _ in range(h)]
    memo = {}
    
    def get_paths_count(row, col, moves_remaining, visited_cells):
        if moves_remaining == 0:
            return 1
        state = (row, col, moves_remaining, tuple(sorted(list(visited_cells))))
        if state in memo:
            return memo[state]
        
        count = 0
        possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in possible_moves:
            next_row, next_col = row + dr, col + dc
            if 1 <= next_row <= h and 1 <= next_col <= w and grid[next_row-1][next_col-1] == '.' and (next_row, next_col) not in visited_cells:
                next_visited_cells = set(visited_cells)
                next_visited_cells.add((next_row, next_col))
                count += get_paths_count(next_row, next_col, moves_remaining - 1, next_visited_cells)
                
        memo[state] = count
        return count
        
    total_paths = 0
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            if grid[i-1][j-1] == '.':
                start_cell = (i, j)
                initial_visited_cells = {start_cell}
                total_paths += get_paths_count(i, j, k, initial_visited_cells)
                
    print(total_paths)

if __name__ == '__main__':
    solve()