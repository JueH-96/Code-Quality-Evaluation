N = int(input())
grid = [input() for _ in range(N)]

count = 0

# For each row i1
for i1 in range(N):
    # For each row i2
    for i2 in range(N):
        # For each row i3
        for i3 in range(N):
            # Skip if not exactly two rows are same
            if len({i1,i2,i3}) != 2:
                continue
                
            # For each column j1
            for j1 in range(N):
                # For each column j2
                for j2 in range(N):
                    # For each column j3
                    for j3 in range(N):
                        # Skip if not exactly two columns are same
                        if len({j1,j2,j3}) != 2:
                            continue
                            
                        # Skip if any cells are same
                        if len({(i1,j1), (i2,j2), (i3,j3)}) != 3:
                            continue
                            
                        # Check if all cells have 'o'
                        if grid[i1][j1] == 'o' and grid[i2][j2] == 'o' and grid[i3][j3] == 'o':
                            count += 1

print(count)