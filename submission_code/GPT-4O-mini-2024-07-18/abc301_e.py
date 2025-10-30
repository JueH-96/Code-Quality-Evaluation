from collections import deque

def bfs(start, goal, grid, H, W, T, candy_positions):
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS queue: (current position, current moves, visited candies)
    queue = deque([(start[0], start[1], 0, 0)])  # (x, y, moves, candy_mask)
    visited = set()
    visited.add((start[0], start[1], 0))  # (x, y, candy_mask)
    
    max_candies = -1
    
    while queue:
        x, y, moves, candy_mask = queue.popleft()
        
        # If we reached the goal and within the move limit
        if (x, y) == goal:
            if moves <= T:
                max_candies = max(max_candies, bin(candy_mask).count('1'))
            continue
        
        # If we exceed the move limit, skip this path
        if moves >= T:
            continue
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                # Determine new candy mask
                new_candy_mask = candy_mask
                if grid[nx][ny] == 'o':
                    candy_index = candy_positions.index((nx, ny))
                    new_candy_mask |= (1 << candy_index)
                
                if (nx, ny, new_candy_mask) not in visited:
                    visited.add((nx, ny, new_candy_mask))
                    queue.append((nx, ny, moves + 1, new_candy_mask))
    
    return max_candies

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    H, W, T = map(int, data[0].split())
    grid = [list(data[i + 1]) for i in range(H)]
    
    start = None
    goal = None
    candy_positions = []
    
    # Locate start, goal, and candy positions
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
            elif grid[i][j] == 'o':
                candy_positions.append((i, j))
    
    # Perform BFS to find the maximum number of candies that can be collected
    result = bfs(start, goal, grid, H, W, T, candy_positions)
    
    print(result)

if __name__ == "__main__":
    main()