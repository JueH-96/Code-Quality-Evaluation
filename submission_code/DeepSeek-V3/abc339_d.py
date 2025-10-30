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
                if len(players) == 2:
                    break
        if len(players) == 2:
            break
    
    # Initialize BFS
    start = (players[0][0], players[0][1], players[1][0], players[1][1])
    visited = set()
    visited.add(start)
    q = deque()
    q.append((start[0], start[1], start[2], start[3], 0))
    
    # Directions: up, down, left, right
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while q:
        x1, y1, x2, y2, steps = q.popleft()
        
        # Check if both players are in the same cell
        if x1 == x2 and y1 == y2:
            print(steps)
            return
        
        for dx, dy in dirs:
            nx1, ny1 = x1 + dx, y1 + dy
            nx2, ny2 = x2 + dx, y2 + dy
            
            # Check if new positions are within bounds and not obstacles
            if nx1 < 0 or nx1 >= N or ny1 < 0 or ny1 >= N or grid[nx1][ny1] == '#':
                nx1, ny1 = x1, y1
            if nx2 < 0 or nx2 >= N or ny2 < 0 or ny2 >= N or grid[nx2][ny2] == '#':
                nx2, ny2 = x2, y2
            
            new_state = (nx1, ny1, nx2, ny2)
            if new_state not in visited:
                visited.add(new_state)
                q.append((nx1, ny1, nx2, ny2, steps + 1))
    
    # If no solution found
    print(-1)

if __name__ == "__main__":
    main()