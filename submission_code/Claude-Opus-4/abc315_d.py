# YOUR CODE HERE
H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(list(input().strip()))

# Create a removed matrix to track which cookies are removed
removed = [[False] * W for _ in range(H)]

while True:
    marked = [[False] * W for _ in range(H)]
    
    # Check rows
    for i in range(H):
        # Find remaining cookies in this row
        remaining = []
        for j in range(W):
            if not removed[i][j]:
                remaining.append((j, grid[i][j]))
        
        # If 2 or more cookies and all same color, mark them
        if len(remaining) >= 2:
            colors = [color for _, color in remaining]
            if all(c == colors[0] for c in colors):
                for j, _ in remaining:
                    marked[i][j] = True
    
    # Check columns
    for j in range(W):
        # Find remaining cookies in this column
        remaining = []
        for i in range(H):
            if not removed[i][j]:
                remaining.append((i, grid[i][j]))
        
        # If 2 or more cookies and all same color, mark them
        if len(remaining) >= 2:
            colors = [color for _, color in remaining]
            if all(c == colors[0] for c in colors):
                for i, _ in remaining:
                    marked[i][j] = True
    
    # Check if any cookies were marked
    any_marked = False
    for i in range(H):
        for j in range(W):
            if marked[i][j]:
                any_marked = True
                removed[i][j] = True
    
    # If no cookies were marked, terminate
    if not any_marked:
        break

# Count remaining cookies
count = 0
for i in range(H):
    for j in range(W):
        if not removed[i][j]:
            count += 1

print(count)