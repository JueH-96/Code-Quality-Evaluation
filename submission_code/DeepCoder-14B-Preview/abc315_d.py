h, w = map(int, input().split())
grid = [list(input().strip()) for _ in range(h)]

while True:
    to_remove = set()
    
    # Process each row
    for i in range(h):
        row = [grid[i][j] for j in range(w) if grid[i][j] is not None]
        if len(row) >= 2 and len(set(row)) == 1:
            for j in range(w):
                if grid[i][j] is not None:
                    to_remove.add((i, j))
    
    # Process each column
    for j in range(w):
        col = [grid[i][j] for i in range(h) if grid[i][j] is not None]
        if len(col) >= 2 and len(set(col)) == 1:
            for i in range(h):
                if grid[i][j] is not None:
                    to_remove.add((i, j))
    
    # If no cells are marked, break the loop
    if not to_remove:
        break
    
    # Remove all marked cells
    for i, j in to_remove:
        grid[i][j] = None

# Count the remaining cells
count = 0
for i in range(h):
    for j in range(w):
        if grid[i][j] is not None:
            count += 1

print(count)