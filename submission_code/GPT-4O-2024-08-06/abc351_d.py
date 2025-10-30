# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = data[2:H+2]
    
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W
    
    def is_adjacent_to_magnet(x, y):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and grid[nx][ny] == '#':
                return True
        return False
    
    def bfs(start_x, start_y):
        from collections import deque
        queue = deque([(start_x, start_y)])
        visited = set()
        visited.add((start_x, start_y))
        count = 0
        
        while queue:
            x, y = queue.popleft()
            count += 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if is_valid(nx, ny) and (nx, ny) not in visited and grid[nx][ny] == '.':
                    if not is_adjacent_to_magnet(nx, ny):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        
        return count
    
    max_degree_of_freedom = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and not is_adjacent_to_magnet(i, j):
                degree_of_freedom = bfs(i, j)
                max_degree_of_freedom = max(max_degree_of_freedom, degree_of_freedom)
    
    print(max_degree_of_freedom)

main()