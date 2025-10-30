def main():
    import sys
    from collections import deque
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    grid = input_data[1:]
    
    # Find the initial positions of the two players
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
    (r1, c1), (r2, c2) = players[0], players[1]
    
    # If they are already in the same cell somehow
    if r1 == r2 and c1 == c2:
        print(0)
        return

    # Directions (row_change, col_change)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # BFS queue: state is (r1, c1, r2, c2, steps)
    queue = deque()
    queue.append((r1, c1, r2, c2, 0))
    
    visited = set()
    visited.add((r1, c1, r2, c2))
    
    def can_move(r, c):
        # Check if (r,c) is within grid and not an obstacle
        return 0 <= r < N and 0 <= c < N and grid[r][c] != '#'
    
    while queue:
        r1, c1, r2, c2, dist = queue.popleft()
        
        for dr, dc in directions:
            # For player 1
            nr1, nc1 = r1 + dr, c1 + dc
            if not can_move(nr1, nc1):
                nr1, nc1 = r1, c1  # stays put
            
            # For player 2
            nr2, nc2 = r2 + dr, c2 + dc
            if not can_move(nr2, nc2):
                nr2, nc2 = r2, c2  # stays put
            
            # Check if players meet
            if nr1 == nr2 and nc1 == nc2:
                print(dist + 1)
                return
            
            if (nr1, nc1, nr2, nc2) not in visited:
                visited.add((nr1, nc1, nr2, nc2))
                queue.append((nr1, nc1, nr2, nc2, dist + 1))
    
    # If not found
    print(-1)

# Do not forget to call main()
if __name__ == "__main__":
    main()