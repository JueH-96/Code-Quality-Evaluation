import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    grid = [sys.stdin.readline().strip() for _ in range(n)]
    
    # Find the two players' positions
    p_positions = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                p_positions.append((i, j))
    
    # Initialize starting positions and normalize
    (r1, c1), (r2, c2) = p_positions
    if (r1, c1) > (r2, c2):
        r1, c1, r2, c2 = r2, c2, r1, c1
    
    queue = deque()
    queue.append((r1, c1, r2, c2, 0))
    visited = set()
    visited.add((r1, c1, r2, c2))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        curr_r1, curr_c1, curr_r2, curr_c2, steps = queue.popleft()
        
        # Check if current positions are the same
        if curr_r1 == curr_r2 and curr_c1 == curr_c2:
            print(steps)
            return
        
        for dr, dc in directions:
            # Calculate new positions for player 1
            nr1, nc1 = curr_r1 + dr, curr_c1 + dc
            if not (0 <= nr1 < n and 0 <= nc1 < n) or grid[nr1][nc1] == '#':
                nr1, nc1 = curr_r1, curr_c1
            
            # Calculate new positions for player 2
            nr2, nc2 = curr_r2 + dr, curr_c2 + dc
            if not (0 <= nr2 < n and 0 <= nc2 < n) or grid[nr2][nc2] == '#':
                nr2, nc2 = curr_r2, curr_c2
            
            # Normalize the new positions
            new_r1, new_c1, new_r2, new_c2 = nr1, nc1, nr2, nc2
            if (new_r1, new_c1) > (new_r2, new_c2):
                new_r1, new_c1, new_r2, new_c2 = new_r2, new_c2, new_r1, new_c1
            
            # Check if they meet at this new state
            if new_r1 == new_r2 and new_c1 == new_c2:
                print(steps + 1)
                return
            
            # Check if this state has been visited
            state = (new_r1, new_c1, new_r2, new_c2)
            if state not in visited:
                visited.add(state)
                queue.append((new_r1, new_c1, new_r2, new_c2, steps + 1))
    
    # If no solution found
    print(-1)

if __name__ == "__main__":
    main()