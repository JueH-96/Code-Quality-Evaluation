import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    grid = [sys.stdin.readline().strip() for _ in range(n)]
    
    players = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                players.append((i, j))
    p1 = players[0]
    p2 = players[1]
    
    if p1 == p2:
        print(0)
        return
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    initial_state = (p1[0], p1[1], p2[0], p2[1])
    visited = set()
    visited.add(initial_state)
    queue = deque()
    queue.append((initial_state, 0))
    
    found = False
    answer = -1
    
    while queue:
        current_state, steps = queue.popleft()
        
        for dr, dc in directions:
            p1r, p1c, p2r, p2c = current_state
            
            # Calculate new position for player 1
            new_p1r = p1r + dr
            new_p1c = p1c + dc
            if 0 <= new_p1r < n and 0 <= new_p1c < n and grid[new_p1r][new_p1c] != '#':
                pass
            else:
                new_p1r, new_p1c = p1r, p1c
            
            # Calculate new position for player 2
            new_p2r = p2r + dr
            new_p2c = p2c + dc
            if 0 <= new_p2r < n and 0 <= new_p2c < n and grid[new_p2r][new_p2c] != '#':
                pass
            else:
                new_p2r, new_p2c = p2r, p2c
            
            # Check if both players meet
            if new_p1r == new_p2r and new_p1c == new_p2c:
                answer = steps + 1
                found = True
                break
            
            new_state = (new_p1r, new_p1c, new_p2r, new_p2c)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, steps + 1))
        
        if found:
            break
    
    print(answer if found else -1)

if __name__ == "__main__":
    main()