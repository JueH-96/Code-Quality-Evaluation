from collections import deque

def solve():
    N, Q = map(int, input().split())
    
    # Initialize colors: cell i has color i (1-indexed)
    colors = [0] + list(range(1, N + 1))  # colors[1] = 1, colors[2] = 2, etc.
    
    for _ in range(Q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:  # Type 1: repaint connected component
            x, c = query[1], query[2]
            original_color = colors[x]
            
            if original_color == c:
                continue  # No change needed
            
            # Find all cells reachable from x with the same color as x
            to_repaint = []
            visited = [False] * (N + 1)
            queue = deque([x])
            visited[x] = True
            
            while queue:
                cell = queue.popleft()
                to_repaint.append(cell)
                
                # Check adjacent cells
                for neighbor in [cell - 1, cell + 1]:
                    if 1 <= neighbor <= N and not visited[neighbor] and colors[neighbor] == original_color:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            # Repaint all found cells
            for cell in to_repaint:
                colors[cell] = c
                
        else:  # Type 2: count cells with color c
            c = query[1]
            count = sum(1 for i in range(1, N + 1) if colors[i] == c)
            print(count)

solve()