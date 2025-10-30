from collections import deque
import sys

def main():
    n = int(sys.stdin.readline())
    grid = [sys.stdin.readline().strip() for _ in range(n)]
    
    # Find the positions of the two players
    players = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                players.append((i, j))
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize BFS
    visited = set()
    start = (players[0][0], players[0][1], players[1][0], players[1][1])
    visited.add(start)
    queue = deque()
    queue.append((players[0][0], players[0][1], players[1][0], players[1][1], 0))
    
    while queue:
        r1, c1, r2, c2, steps = queue.popleft()
        
        # Check if they have met
        if r1 == r2 and c1 == c2:
            print(steps)
            return
        
        # Try each direction
        for dr, dc in directions:
            # Compute new positions for player 1
            nr1, nc1 = r1 + dr, c1 + dc
            if 0 <= nr1 < n and 0 <= nc1 < n and grid[nr1][nc1] != '#':
                pass  # Move to new position
            else:
                nr1, nc1 = r1, c1  # Stay in place
            
            # Compute new positions for player 2
            nr2, nc2 = r2 + dr, c2 + dc
            if 0 <= nr2 < n and 0 <= nc2 < n and grid[nr2][nc2] != '#':
                pass
            else:
                nr2, nc2 = r2, c2
            
            # Check if this state has been visited
            new_state = (nr1, nc1, nr2, nc2)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((nr1, nc1, nr2, nc2, steps + 1))
    
    # If queue is empty and no solution found
    print(-1)

if __name__ == "__main__":
    main()