import sys
data = sys.stdin.read().split()
index = 0
T = int(data[index])
index += 1
for test in range(T):
    H = int(data[index])
    W = int(data[index+1])
    index += 2
    grid = []
    for i in range(H):
        s = data[index]
        grid.append(s)
        index += 1
    # Check if all rows have even number of 'A'
    row_even = True
    for row in grid:
        a_count = row.count('A')
        if a_count % 2 == 1:
            row_even = False
            break
    if not row_even:
        print(0)
        continue
    # Check if all columns have even number of 'A'
    col_even = True
    for j in range(W):
        a_count = 0
        for i in range(H):
            if grid[i][j] == 'A':
                a_count += 1
        if a_count % 2 == 1:
            col_even = False
            break
    if not col_even:
        print(0)
        continue
    # If both conditions are satisfied, output 2
    print(2)