N = int(input())

grid = []
for r in range(1, N + 1):
    row = ""
    for c in range(1, N + 1):
        # Find the last operation that affects cell (r,c)
        last_i = min(min(r, c), N + 1 - max(r, c))
        
        # Check if this operation is valid (i <= j)
        j = N + 1 - last_i
        if last_i <= j:
            if last_i % 2 == 1:  # odd i -> black
                row += "#"
            else:  # even i -> white
                row += "."
        else:
            # This shouldn't happen based on problem statement
            row += "?"
    grid.append(row)

for row in grid:
    print(row)