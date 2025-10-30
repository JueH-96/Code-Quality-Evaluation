from collections import deque

def main():
    n = int(input())
    grid = [input().strip() for _ in range(n)]
    
    positions = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                positions.append((i, j))
    
    x1, y1 = positions[0]
    x2, y2 = positions[1]
    
    if x1 == x2 and y1 == y2:
        print(0)
        return
    
    visited = set()
    queue = deque()
    initial_state = (x1, y1, x2, y2)
    queue.append((x1, y1, x2, y2, 0))
    visited.add(initial_state)
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        x1_curr, y1_curr, x2_curr, y2_curr, steps = queue.popleft()
        
        if (x1_curr, y1_curr) == (x2_curr, y2_curr):
            print(steps)
            return
        
        for dx, dy in directions:
            # Compute new position for player 1
            new_x1 = x1_curr + dx
            new_y1 = y1_curr + dy
            if 0 <= new_x1 < n and 0 <= new_y1 < n:
                if grid[new_x1][new_y1] != '#':
                    new_p1 = (new_x1, new_y1)
                else:
                    new_p1 = (x1_curr, y1_curr)
            else:
                new_p1 = (x1_curr, y1_curr)
            
            # Compute new position for player 2
            new_x2 = x2_curr + dx
            new_y2 = y2_curr + dy
            if 0 <= new_x2 < n and 0 <= new_y2 < n:
                if grid[new_x2][new_y2] != '#':
                    new_p2 = (new_x2, new_y2)
                else:
                    new_p2 = (x2_curr, y2_curr)
            else:
                new_p2 = (x2_curr, y2_curr)
            
            new_state = (new_p1[0], new_p1[1], new_p2[0], new_p2[1])
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_p1[0], new_p1[1], new_p2[0], new_p2[1], steps + 1))
    
    print(-1)

if __name__ == "__main__":
    main()