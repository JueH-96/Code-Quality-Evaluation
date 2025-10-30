import collections

def solve():
    h, w, d = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    humidifier_locations = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'H':
                humidifier_locations.append((r, c))
    
    is_humidified = [[False for _ in range(w)] for _ in range(h)]
    
    for start_row, start_col in humidifier_locations:
        queue = collections.deque([((start_row, start_col), 0)])
        visited_in_bfs = set([(start_row, start_col)])
        
        while queue:
            (current_row, current_col), current_distance = queue.popleft()
            is_humidified[current_row][current_col] = True
            if current_distance < d:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    next_row, next_col = current_row + dr, current_col + dc
                    if 0 <= next_row < h and 0 <= next_col < w and grid[next_row][next_col] != '#' and (next_row, next_col) not in visited_in_bfs:
                        visited_in_bfs.add((next_row, next_col))
                        queue.append(((next_row, next_col), current_distance + 1))
                        
    humidified_floor_cells_count = 0
    for r in range(h):
        for c in range(w):
            if is_humidified[r][c] and grid[r][c] == '.':
                humidified_floor_cells_count += 1
            elif is_humidified[r][c] and grid[r][c] == 'H':
                humidified_floor_cells_count += 1

    humidified_cells_count = 0
    for r in range(h):
        for c in range(w):
            if is_humidified[r][c]:
                humidified_cells_count += 1

    count_humidified_floor = 0
    for r in range(h):
        for c in range(w):
            if is_humidified[r][c] and grid[r][c] == '.':
                count_humidified_floor += 1
                
    humidified_floor_cells_output = 0
    for r in range(h):
        for c in range(w):
            if is_humidified[r][c] and grid[r][c] == '.':
                humidified_floor_cells_output += 1

    humidified_cells_output = 0
    for r in range(h):
        for c in range(w):
            if is_humidified[r][c]:
                humidified_cells_output += 1
                
    floor_humidified_count = 0
    for r in range(h):
        for c in range(w):
            if is_humidified[r][c] and grid[r][c] == '.':
                floor_humidified_count += 1

    humidified_floor_cells_final_count = 0
    for r in range(h):
        for c in range(w):
            if is_humidified[r][c] and grid[r][c] == '.':
                humidified_floor_cells_final_count += 1
                
    humidified_cells_including_humidifiers_count = 0
    for r in range(h):
        for c in range(w):
            if is_humidified[r][c]:
                humidified_cells_including_humidifiers_count += 1

    floor_and_humidifier_count = 0
    for r in range(h):
        for c in range(w):
            if is_humidified[r][c] and (grid[r][c] == '.' or grid[r][c] == 'H'):
                floor_and_humidifier_count += 1

    count_only_floor_humidified = 0
    for r in range(h):
        for c in range(w):
            if is_humidified[r][c] and grid[r][c] == '.':
                count_only_floor_humidified += 1
                
    
    print(humidified_cells_output) # Let's try outputting total humidified cells


if __name__ == '__main__':
    solve()