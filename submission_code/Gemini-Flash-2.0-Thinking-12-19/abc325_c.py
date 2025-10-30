def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    sensor_locations = []
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '#':
                sensor_locations.append((i, j))
    
    visited_sensors = set()
    group_count = 0
    
    def is_valid(r, c):
        return 0 <= r < h and 0 <= c < w
        
    def get_adjacent_sensors(r, c):
        adjacent_list = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if is_valid(nr, nc) and grid[nr][nc] == '#':
                    adjacent_list.append((nr, nc))
        return adjacent_list
        
    def dfs(location):
        if location in visited_sensors:
            return
        visited_sensors.add(location)
        r, c = location
        for neighbor_loc in get_adjacent_sensors(r, c):
            dfs(neighbor_loc)
            
    for sensor_loc in sensor_locations:
        if sensor_loc not in visited_sensors:
            group_count += 1
            dfs(sensor_loc)
            
    print(group_count)

if __name__ == '__main__':
    solve()