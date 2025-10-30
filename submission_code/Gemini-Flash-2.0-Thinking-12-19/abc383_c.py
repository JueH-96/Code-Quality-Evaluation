import collections

def solve():
    h, w, d = map(int, input().split())
    grid = [input() for _ in range(h)]
    humidifier_locations = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'H':
                humidifier_locations.append((r, c))
    
    all_humidified_locations = set()
    for start_row, start_col in humidifier_locations:
        reachable_cells = set([(start_row, start_col)])
        q = collections.deque([((start_row, start_col), 0)])
        local_visited = set([(start_row, start_col)])
        while q:
            (current_row, current_col), distance = q.popleft()
            if distance < d:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    next_row, next_col = current_row + dr, current_col + dc
                    if 0 <= next_row < h and 0 <= next_col < w and grid[next_row][next_col] != '#' and (next_row, next_col) not in local_visited:
                        reachable_cells.add((next_row, next_col))
                        local_visited.add((next_row, next_col))
                        q.append(((next_row, next_col), distance + 1))
        all_humidified_locations.update(reachable_cells)
        
    humidified_floor_cells_count = 0
    for r, c in all_humidified_locations:
        if grid[r][c] == '.' or grid[r][c] == 'H':
            if grid[r][c] == '.':
                humidified_floor_cells_count += 1
            elif grid[r][c] == 'H':
                humidified_floor_cells_count += 1

    humidified_floor_cells_count = 0
    for r, c in all_humidified_locations:
        if grid[r][c] == '.':
            humidified_floor_cells_count += 1
            
    print(humidified_floor_cells_count)

if __name__ == '__main__':
    solve()