n, m = map(int, input().split())
grid = [input().strip() for _ in range(n)]

max_i = n - 9 + 1
max_j = m - 9 + 1

result = []

for i in range(1, max_i + 1):
    for j in range(1, max_j + 1):
        valid = True
        
        # Check top-left 3x3 is all '#'
        for dx in range(3):
            if not valid:
                break
            for dy in range(3):
                x = i - 1 + dx
                y = j - 1 + dy
                if grid[x][y] != '#':
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Check bottom-right 3x3 is all '#'
        for dx in range(6, 9):
            if not valid:
                break
            for dy in range(6, 9):
                x = i - 1 + dx
                y = j - 1 + dy
                if grid[x][y] != '#':
                    valid = False
                    break
        
        if not valid:
            continue
        
        # Check adjacent cells for top-left 3x3
        adjacent_tl = [(0,3), (1,3), (2,3), (3,0), (3,1), (3,2), (3,3)]
        for dx, dy in adjacent_tl:
            x = i - 1 + dx
            y = j - 1 + dy
            if grid[x][y] != '.':
                valid = False
                break
        
        if not valid:
            continue
        
        # Check adjacent cells for bottom-right 3x3
        adjacent_br = [(5,5), (5,6), (5,7), (5,8), (6,5), (7,5), (8,5)]
        for dx, dy in adjacent_br:
            x = i - 1 + dx
            y = j - 1 + dy
            if grid[x][y] != '.':
                valid = False
                break
        
        if valid:
            result.append((i, j))

# Sort the result lexicographically
result.sort()

# Output each pair
for pair in result:
    print(pair[0], pair[1])