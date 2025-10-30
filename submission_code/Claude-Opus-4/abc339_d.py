from collections import deque

def solve():
    N = int(input())
    grid = []
    for _ in range(N):
        grid.append(input().strip())
    
    # Find initial positions of the two players
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
    
    p1_start = players[0]
    p2_start = players[1]
    
    # BFS to find minimum moves
    # State: (p1_row, p1_col, p2_row, p2_col)
    queue = deque([(p1_start[0], p1_start[1], p2_start[0], p2_start[1], 0)])
    visited = set()
    visited.add((p1_start[0], p1_start[1], p2_start[0], p2_start[1]))
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        p1_r, p1_c, p2_r, p2_c, moves = queue.popleft()
        
        # Check if players are at the same position
        if p1_r == p2_r and p1_c == p2_c:
            print(moves)
            return
        
        # Try all four directions
        for dr, dc in directions:
            # Calculate new positions
            new_p1_r, new_p1_c = p1_r + dr, p1_c + dc
            new_p2_r, new_p2_c = p2_r + dr, p2_c + dc
            
            # Check if player 1 can move
            if (0 <= new_p1_r < N and 0 <= new_p1_c < N and 
                grid[new_p1_r][new_p1_c] != '#'):
                # Player 1 can move
                pass
            else:
                # Player 1 cannot move, stays in place
                new_p1_r, new_p1_c = p1_r, p1_c
            
            # Check if player 2 can move
            if (0 <= new_p2_r < N and 0 <= new_p2_c < N and 
                grid[new_p2_r][new_p2_c] != '#'):
                # Player 2 can move
                pass
            else:
                # Player 2 cannot move, stays in place
                new_p2_r, new_p2_c = p2_r, p2_c
            
            # Check if this state has been visited
            state = (new_p1_r, new_p1_c, new_p2_r, new_p2_c)
            if state not in visited:
                visited.add(state)
                queue.append((new_p1_r, new_p1_c, new_p2_r, new_p2_c, moves + 1))
    
    # If we exit the loop without finding a solution
    print(-1)

solve()