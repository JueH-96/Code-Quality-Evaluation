def main():
    import sys
    from collections import deque

    data = sys.stdin.read().splitlines()
    if not data:
        return

    N = int(data[0].strip())
    grid = []
    players = []
    for i in range(N):
        line = data[i+1].rstrip("
")
        grid.append(line)
        for j, char in enumerate(line):
            if char == 'P':
                players.append((i, j))
    
    # There are exactly two players in distinct empty cells.
    (p1_i, p1_j), (p2_i, p2_j) = players[0], players[1]

    # Define the four possible move directions: up, down, left, right.
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    # Helper function to determine if a given cell is within the grid bounds
    # and is not an obstacle.
    def can_move(i, j):
        if 0 <= i < N and 0 <= j < N and grid[i][j] != '#':
            return True
        return False
    
    # We'll perform BFS over the state space.
    # A state is represented by (i1, j1, i2, j2) for the positions of Player1 and Player2.
    start = (p1_i, p1_j, p2_i, p2_j)
    visited = set([start])
    dq = deque()
    dq.append((start, 0))
    
    while dq:
        (i1, j1, i2, j2), moves = dq.popleft()
        
        # Check if players have met.
        if i1 == i2 and j1 == j2:
            sys.stdout.write(str(moves))
            return

        # Try all four directions.
        for di, dj in directions:
            # For Player 1
            new_i1, new_j1 = i1, j1  # Assume player stands still.
            next_i1, next_j1 = i1 + di, j1 + dj
            if can_move(next_i1, next_j1):
                new_i1, new_j1 = next_i1, next_j1

            # For Player 2
            new_i2, new_j2 = i2, j2
            next_i2, next_j2 = i2 + di, j2 + dj
            if can_move(next_i2, next_j2):
                new_i2, new_j2 = next_i2, next_j2
            
            new_state = (new_i1, new_j1, new_i2, new_j2)
            if new_state in visited:
                continue
            visited.add(new_state)
            if new_i1 == new_i2 and new_j1 == new_j2:
                sys.stdout.write(str(moves + 1))
                return
            dq.append((new_state, moves + 1))
    
    # If we exit the BFS loop without meeting, it's impossible.
    sys.stdout.write("-1")

if __name__ == '__main__':
    main()