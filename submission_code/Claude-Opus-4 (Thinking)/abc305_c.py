H, W = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input().strip())

# Try all possible rectangles with height >= 2 and width >= 2
for a in range(H):
    for b in range(a + 1, H):  # b > a ensures height >= 2
        for c in range(W):
            for d in range(c + 1, W):  # d > c ensures width >= 2
                # Check this rectangle
                empty_cells = []
                has_cookie_outside = False
                
                for i in range(H):
                    for j in range(W):
                        if a <= i <= b and c <= j <= d:
                            # Inside rectangle
                            if grid[i][j] == '.':
                                empty_cells.append((i, j))
                        else:
                            # Outside rectangle
                            if grid[i][j] == '#':
                                has_cookie_outside = True
                                break
                    if has_cookie_outside:
                        break
                
                # Check if this is the valid rectangle
                if not has_cookie_outside and len(empty_cells) == 1:
                    # Found it!
                    print(empty_cells[0][0] + 1, empty_cells[0][1] + 1)
                    exit()