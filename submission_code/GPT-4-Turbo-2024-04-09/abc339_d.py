def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    grid = data[1:]
    
    # Directions: (dy, dx)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Find the positions of the two players
    players = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'P':
                players.append((i, j))
    
    from collections import deque
    
    # BFS setup
    queue = deque()
    queue.append((players[0][0], players[0][1], players[1][0], players[1][1], 0))
    visited = set()
    visited.add((players[0][0], players[0][1], players[1][0], players[1][1]))
    
    # BFS to find the minimum moves
    while queue:
        y1, x1, y2, x2, moves = queue.popleft()
        
        # If both players meet
        if y1 == y2 and x1 == x2:
            print(moves)
            return
        
        # Try all directions
        for dy, dx in directions:
            ny1, nx1 = y1 + dy, x1 + dx
            ny2, nx2 = y2 + dy, x2 + dx
            
            # Check boundaries and obstacles for player 1
            if not (0 <= ny1 < N and 0 <= nx1 < N and grid[ny1][nx1] != '#'):
                ny1, nx1 = y1, x1
            
            # Check boundaries and obstacles for player 2
            if not (0 <= ny2 < N and 0 <= nx2 < N and grid[ny2][nx2] != '#'):
                ny2, nx2 = y2, x2
            
            # If new state is not visited, add to queue
            if (ny1, nx1, ny2, nx2) not in visited:
                visited.add((ny1, nx1, ny2, nx2))
                queue.append((ny1, nx1, ny2, nx2, moves + 1))
    
    # If no solution found
    print(-1)