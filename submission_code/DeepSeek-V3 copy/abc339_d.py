from collections import deque

def main():
    N = int(input())
    grid = [input().strip() for _ in range(N)]
    
    # Find the positions of the two players
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
    (x1, y1), (x2, y2) = players
    
    # Initialize BFS
    visited = set()
    queue = deque()
    queue.append((x1, y1, x2, y2, 0))
    visited.add((x1, y1, x2, y2))
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        cx1, cy1, cx2, cy2, steps = queue.popleft()
        
        # Check if both players are in the same cell
        if cx1 == cx2 and cy1 == cy2:
            print(steps)
            return
        
        for dx, dy in directions:
            nx1, ny1 = cx1 + dx, cy1 + dy
            nx2, ny2 = cx2 + dx, cy2 + dy
            
            # Check if new positions are within bounds and not obstacles
            if nx1 < 0 or nx1 >= N or ny1 < 0 or ny1 >= N or grid[nx1][ny1] == '#':
                nx1, ny1 = cx1, cy1
            if nx2 < 0 or nx2 >= N or ny2 < 0 or ny2 >= N or grid[nx2][ny2] == '#':
                nx2, ny2 = cx2, cy2
            
            # Check if this state has been visited
            if (nx1, ny1, nx2, ny2) not in visited:
                visited.add((nx1, ny1, nx2, ny2))
                queue.append((nx1, ny1, nx2, ny2, steps + 1))
    
    # If no solution found
    print(-1)

if __name__ == "__main__":
    main()