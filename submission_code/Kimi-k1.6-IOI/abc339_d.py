from collections import deque
import sys

def main():
    n = int(sys.stdin.readline())
    grid = [sys.stdin.readline().strip() for _ in range(n)]
    
    # Find the initial positions of the two players
    players = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                players.append((i, j))
    p1, p2 = players
    
    # Canonicalize the initial state
    if p1 > p2:
        p1, p2 = p2, p1
    initial_state = (p1[0], p1[1], p2[0], p2[1])
    
    visited = set()
    visited.add(initial_state)
    queue = deque()
    queue.append((p1[0], p1[1], p2[0], p2[1], 0))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    found = False
    answer = -1
    
    while queue:
        x1, y1, x2, y2, steps = queue.popleft()
        
        # Check if both players are in the same cell
        if x1 == x2 and y1 == y2:
            answer = steps
            found = True
            break
        
        # Generate all four possible directions
        for dx, dy in directions:
            # Compute new positions for player 1
            nx1 = x1 + dx
            ny1 = y1 + dy
            if 0 <= nx1 < n and 0 <= ny1 < n and grid[nx1][ny1] != '#':
                new_p1 = (nx1, ny1)
            else:
                new_p1 = (x1, y1)
            
            # Compute new positions for player 2
            nx2 = x2 + dx
            ny2 = y2 + dy
            if 0 <= nx2 < n and 0 <= ny2 < n and grid[nx2][ny2] != '#':
                new_p2 = (nx2, ny2)
            else:
                new_p2 = (x2, y2)
            
            # Canonicalize the new state
            pos1 = new_p1
            pos2 = new_p2
            if pos1 > pos2:
                pos1, pos2 = pos2, pos1
            new_state = (pos1[0], pos1[1], pos2[0], pos2[1])
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((pos1[0], pos1[1], pos2[0], pos2[1], steps + 1))
    
    print(answer if found else -1)

if __name__ == "__main__":
    main()