import sys
from collections import deque

def main():
    n = int(sys.stdin.readline())
    grid = [sys.stdin.readline().strip() for _ in range(n)]
    
    positions = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'P':
                positions.append((i, j))
    
    x1, y1 = positions[0]
    x2, y2 = positions[1]
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    visited = set()
    initial_state = ((x1, y1), (x2, y2))
    queue = deque()
    queue.append((initial_state, 0))
    visited.add(initial_state)
    
    while queue:
        current_state, steps = queue.popleft()
        if current_state[0] == current_state[1]:
            print(steps)
            return
        
        for dr, dc in directions:
            new_x1 = x1 + dr
            new_y1 = y1 + dc
            if 0 <= new_x1 < n and 0 <= new_y1 < n:
                if grid[new_x1][new_y1] != '#':
                    new_x1_final = new_x1
                    new_y1_final = new_y1
                else:
                    new_x1_final = x1
                    new_y1_final = y1
            else:
                new_x1_final = x1
                new_y1_final = y1
            
            new_x2 = x2 + dr
            new_y2 = y2 + dc
            if 0 <= new_x2 < n and 0 <= new_y2 < n:
                if grid[new_x2][new_y2] != '#':
                    new_x2_final = new_x2
                    new_y2_final = new_y2
                else:
                    new_x2_final = x2
                    new_y2_final = y2
            else:
                new_x2_final = x2
                new_y2_final = y2
            
            new_state = ((new_x1_final, new_y1_final), (new_x2_final, new_y2_final))
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, steps + 1))
    
    print(-1)

if __name__ == '__main__':
    main()