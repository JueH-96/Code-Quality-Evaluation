def solve():
    N, M = map(int, input().split())
    grid = [input() for _ in range(N)]
    
    # Keep track of visited squares
    visited = set()
    visited.add((1, 1))  # 0-based coordinates
    
    # Queue for positions to explore from
    positions = {(1, 1)}  # Start at (2,2) in 0-based coordinates
    
    # Try moving from each position we can reach
    while positions:
        curr_pos = positions.pop()
        
        # Try each direction
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            x, y = curr_pos
            
            # Keep moving in this direction until hitting a rock
            while True:
                next_x = x + dx
                next_y = y + dy
                
                # Check if next position is rock
                if grid[next_x][next_y] == '#':
                    # If we moved at least one step, add current position
                    if (x, y) != curr_pos:
                        positions.add((x, y))
                    break
                    
                # Mark as visited and continue moving
                visited.add((next_x, next_y))
                x, y = next_x, next_y
    
    print(len(visited))

solve()