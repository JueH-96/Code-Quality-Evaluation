from collections import deque

def find_players(grid):
    players = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 'P':
                players.append((i, j))
    return players

def is_valid(x, y, n):
    return 0 <= x < n and 0 <= y < n

def solve():
    n = int(input())
    grid = [input() for _ in range(n)]
    
    # Find initial positions of players
    p1, p2 = find_players(grid)
    
    # BFS queue stores (pos1, pos2, moves)
    q = deque([(p1, p2, 0)])
    # visited set stores (pos1, pos2)
    visited = {(p1, p2)}
    
    # Possible directions: up, down, left, right
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    
    while q:
        (x1, y1), (x2, y2), moves = q.popleft()
        
        # If players are at same position, return moves
        if (x1, y1) == (x2, y2):
            return moves
            
        # Try all directions
        for d in range(4):
            nx1, ny1 = x1 + dx[d], y1 + dy[d]
            nx2, ny2 = x2 + dx[d], y2 + dy[d]
            
            # Check if new positions are valid for player 1
            if not is_valid(nx1, ny1, n) or grid[nx1][ny1] == '#':
                nx1, ny1 = x1, y1
                
            # Check if new positions are valid for player 2
            if not is_valid(nx2, ny2, n) or grid[nx2][ny2] == '#':
                nx2, ny2 = x2, y2
                
            new_state = ((nx1, ny1), (nx2, ny2))
            if new_state not in visited:
                visited.add(new_state)
                q.append((new_state[0], new_state[1], moves + 1))
    
    return -1

print(solve())