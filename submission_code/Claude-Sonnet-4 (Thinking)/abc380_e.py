from collections import deque

n, q = map(int, input().split())

# Initialize colors
colors = [i for i in range(n + 1)]  # colors[i] = color of cell i, 1-indexed

for _ in range(q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Type 1: repaint
        x, c = query[1], query[2]
        original_color = colors[x]
        
        if original_color == c:
            continue  # No need to repaint
        
        # BFS to find all connected cells with the same color
        to_repaint = []
        visited = [False] * (n + 1)
        queue = deque([x])
        visited[x] = True
        
        while queue:
            cell = queue.popleft()
            to_repaint.append(cell)
            
            # Check adjacent cells
            for neighbor in [cell - 1, cell + 1]:
                if 1 <= neighbor <= n and not visited[neighbor] and colors[neighbor] == original_color:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        # Repaint all connected cells
        for cell in to_repaint:
            colors[cell] = c
    
    else:
        # Type 2: count
        c = query[1]
        count = sum(1 for i in range(1, n + 1) if colors[i] == c)
        print(count)