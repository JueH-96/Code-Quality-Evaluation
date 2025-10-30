import sys
from collections import deque

def main():
    N = int(sys.stdin.readline())
    grid = []
    players = []
    for i in range(N):
        line = sys.stdin.readline().strip()
        grid.append(line)
        for j in range(N):
            if line[j] == 'P':
                players.append((i, j))
    p1 = players[0]
    p2 = players[1]
    
    # Sort the initial state to avoid duplicate states
    if p1 <= p2:
        initial_state = (p1[0], p1[1], p2[0], p2[1])
    else:
        initial_state = (p2[0], p2[1], p1[0], p1[1])
    
    visited = set()
    visited.add(initial_state)
    queue = deque()
    queue.append((initial_state[0], initial_state[1], initial_state[2], initial_state[3], 0))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r1, c1, r2, c2, steps = queue.popleft()
        
        # Check if both players are at the same position
        if r1 == r2 and c1 == c2:
            print(steps)
            return
        
        for dr, dc in directions:
            # Calculate new positions for player 1
            nr1 = r1 + dr
            nc1 = c1 + dc
            if 0 <= nr1 < N and 0 <= nc1 < N and grid[nr1][nc1] != '#':
                new_r1, new_c1 = nr1, nc1
            else:
                new_r1, new_c1 = r1, c1
            
            # Calculate new positions for player 2
            nr2 = r2 + dr
            nc2 = c2 + dc
            if 0 <= nr2 < N and 0 <= nc2 < N and grid[nr2][nc2] != '#':
                new_r2, new_c2 = nr2, nc2
            else:
                new_r2, new_c2 = r2, c2
            
            # Create new state with sorted positions
            pos1 = (new_r1, new_c1)
            pos2 = (new_r2, new_c2)
            if pos1 <= pos2:
                new_state = (pos1[0], pos1[1], pos2[0], pos2[1])
            else:
                new_state = (pos2[0], pos2[1], pos1[0], pos1[1])
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state[0], new_state[1], new_state[2], new_state[3], steps + 1))
    
    # If no meeting point found
    print(-1)

if __name__ == '__main__':
    main()