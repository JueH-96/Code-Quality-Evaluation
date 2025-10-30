import sys
from collections import deque

def main():
    N = int(sys.stdin.readline().strip())
    grid = []
    for _ in range(N):
        line = sys.stdin.readline().strip()
        grid.append(line)
    
    # Find the two P positions
    p1 = None
    p2 = None
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                if p1 is None:
                    p1 = (i, j)
                else:
                    p2 = (i, j)
    
    # Initial state sorted
    sorted_pos = sorted([p1, p2])
    initial_r1, initial_c1 = sorted_pos[0]
    initial_r2, initial_c2 = sorted_pos[1]
    
    # Directions: up, down, left, right
    directions = [ (-1,0), (1,0), (0,-1), (0,1) ]
    
    visited = set()
    queue = deque()
    
    initial_state = (initial_r1, initial_c1, initial_r2, initial_c2)
    visited.add(initial_state)
    queue.append( (initial_r1, initial_c1, initial_r2, initial_c2, 0) )
    
    found = False
    while queue:
        r1, c1, r2, c2, steps = queue.popleft()
        
        if r1 == r2 and c1 == c2:
            print(steps)
            found = True
            break
        
        for dr, dc in directions:
            # Compute new positions for both players
            # Player 1's new position
            nr1 = r1 + dr
            nc1 = c1 + dc
            if 0 <= nr1 < N and 0 <= nc1 < N and grid[nr1][nc1] != '#':
                np1r, np1c = nr1, nc1
            else:
                np1r, np1c = r1, c1
            
            # Player 2's new position
            nr2 = r2 + dr
            nc2 = c2 + dc
            if 0 <= nr2 < N and 0 <= nc2 < N and grid[nr2][nc2] != '#':
                np2r, np2c = nr2, nc2
            else:
                np2r, np2c = r2, c2
            
            # Create new state sorted
            new_pos1 = (np1r, np1c)
            new_pos2 = (np2r, np2c)
            sorted_new = sorted([new_pos1, new_pos2])
            new_state = (sorted_new[0][0], sorted_new[0][1], sorted_new[1][0], sorted_new[1][1])
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append( (new_state[0], new_state[1], new_state[2], new_state[3], steps + 1) )
    
    if not found:
        print(-1)

if __name__ == "__main__":
    main()