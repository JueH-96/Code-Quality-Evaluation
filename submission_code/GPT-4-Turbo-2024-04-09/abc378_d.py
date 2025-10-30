def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    K = int(data[2])
    grid = [data[i + 3] for i in range(H)]
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W and grid[x][y] == '.'
    
    from collections import deque
    
    # Count paths starting from each cell
    total_paths = 0
    
    for start_x in range(H):
        for start_y in range(W):
            if grid[start_x][start_y] == '#':
                continue
            
            # BFS to find all paths of length K starting from (start_x, start_y)
            queue = deque([(start_x, start_y, 0, {(start_x, start_y)})])  # (x, y, steps, visited_set)
            while queue:
                x, y, steps, visited = queue.popleft()
                
                if steps == K:
                    total_paths += 1
                    continue
                
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if is_valid(nx, ny) and (nx, ny) not in visited:
                        new_visited = set(visited)
                        new_visited.add((nx, ny))
                        queue.append((nx, ny, steps + 1, new_visited))
    
    print(total_paths)