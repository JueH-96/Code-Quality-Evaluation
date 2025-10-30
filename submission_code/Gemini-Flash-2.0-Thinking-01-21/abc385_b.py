def solve():
    h, w, start_row, start_col = map(int, input().split())
    grid_lines = [input() for _ in range(h)]
    moves = input()
    
    grid = []
    for line in grid_lines:
        grid.append(list(line))
        
    current_row = start_row - 1
    current_col = start_col - 1
    visited_houses = set()
    
    if grid[current_row][current_col] == '@':
        visited_houses.add((current_row, current_col))
        
    for move in moves:
        next_row, next_col = current_row, current_col
        if move == 'U':
            next_row = current_row - 1
        elif move == 'D':
            next_row = current_row + 1
        elif move == 'L':
            next_col = current_col - 1
        elif move == 'R':
            next_col = current_col + 1
            
        if 0 <= next_row < h and 0 <= next_col < w and grid[next_row][next_col] != '#':
            current_row, current_col = next_row, next_col
            
        if grid[current_row][current_col] == '@':
            visited_houses.add((current_row, current_col))
            
    print(current_row + 1, current_col + 1, len(visited_houses))

if __name__ == '__main__':
    solve()