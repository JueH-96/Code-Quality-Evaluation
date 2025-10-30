import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    grid = [sys.stdin.readline().strip() for _ in range(N)]
    
    # Find the initial positions of the two players
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
    if len(players) != 2:
        print(-1)
        return
    (x1, y1), (x2, y2) = players
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS initialization
    visited = set()
    queue = deque()
    initial_state = (x1, y1, x2, y2)
    visited.add(initial_state)
    queue.append((x1, y1, x2, y2, 0))
    
    found = False
    while queue:
        current = queue.popleft()
        cx1, cy1, cx2, cy2, steps = current
        
        # Check if they meet
        if cx1 == cx2 and cy1 == cy2:
            print(steps)
            found = True
            break
        
        # Generate next states for all directions
        for dx, dy in directions:
            # Calculate new positions for player 1
            nx1, ny1 = cx1 + dx, cy1 + dy
            if 0 <= nx1 < N and 0 <= ny1 < N and grid[nx1][ny1] != '#':
                new_p1 = (nx1, ny1)
            else:
                new_p1 = (cx1, cy1)
            
            # Calculate new positions for player 2
            nx2, ny2 = cx2 + dx, cy2 + dy
            if 0 <= nx2 < N and 0 <= ny2 < N and grid[nx2][ny2] != '#':
                new_p2 = (nx2, ny2)
            else:
                new_p2 = (cx2, cy2)
            
            new_state = (new_p1[0], new_p1[1], new_p2[0], new_p2[1])
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_p1[0], new_p1[1], new_p2[0], new_p2[1], steps + 1))
    
    if not found:
        print(-1)

if __name__ == "__main__":
    main()