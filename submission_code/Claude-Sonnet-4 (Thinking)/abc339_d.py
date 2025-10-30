from collections import deque

def solve():
    n = int(input())
    grid = []
    for _ in range(n):
        grid.append(input().strip())

    # Find player positions
    players = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                players.append((i, j))

    p1_start = players[0]
    p2_start = players[1]

    # BFS
    queue = deque([(p1_start[0], p1_start[1], p2_start[0], p2_start[1], 0)])
    visited = set()
    visited.add((p1_start[0], p1_start[1], p2_start[0], p2_start[1]))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    while queue:
        p1_r, p1_c, p2_r, p2_c, moves = queue.popleft()
        
        # Check if both players are at the same position
        if p1_r == p2_r and p1_c == p2_c:
            return moves
        
        # Try all 4 directions
        for dr, dc in directions:
            # Player 1's new position
            new_p1_r, new_p1_c = p1_r + dr, p1_c + dc
            if (0 <= new_p1_r < n and 0 <= new_p1_c < n and 
                grid[new_p1_r][new_p1_c] != '#'):
                # Player 1 can move
                pass
            else:
                # Player 1 can't move, stays in place
                new_p1_r, new_p1_c = p1_r, p1_c
            
            # Player 2's new position
            new_p2_r, new_p2_c = p2_r + dr, p2_c + dc
            if (0 <= new_p2_r < n and 0 <= new_p2_c < n and 
                grid[new_p2_r][new_p2_c] != '#'):
                # Player 2 can move
                pass
            else:
                # Player 2 can't move, stays in place
                new_p2_r, new_p2_c = p2_r, p2_c
            
            new_state = (new_p1_r, new_p1_c, new_p2_r, new_p2_c)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_p1_r, new_p1_c, new_p2_r, new_p2_c, moves + 1))

    return -1

print(solve())