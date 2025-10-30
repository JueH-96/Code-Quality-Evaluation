# Read input
h_a, w_a = map(int, input().split())
a = [input().strip() for _ in range(h_a)]
h_b, w_b = map(int, input().split())
b = [input().strip() for _ in range(h_b)]
h_x, w_x = map(int, input().split())
x = [input().strip() for _ in range(h_x)]

# Collect black squares for A and B
a_black = [ (i, j) for i in range(h_a) for j in range(w_a) if a[i][j] == '#' ]
b_black = [ (i, j) for i in range(h_b) for j in range(w_b) if b[i][j] == '#' ]

# If either A or B has no black squares, it's impossible (though problem states each has at least one)
if not a_black or not b_black:
    print("No")
    exit()

# Iterate over possible top-left positions of the cut-out window (cx, cy)
# We use a reasonable range to cover possible positions
for cx in range(-200, 200 + h_x):
    for cy in range(-200, 200 + w_x):
        # Calculate possible x_a range for A
        a_black_rows = [i for i, j in a_black]
        a_black_cols = [j for i, j in a_black]
        x_a_min = max([cx - i for i, j in a_black])
        x_a_max = min([(cx + h_x - 1) - i for i, j in a_black])
        if x_a_min > x_a_max:
            continue
        
        y_a_min = max([cy - j for i, j in a_black])
        y_a_max = min([(cy + w_x - 1) - j for i, j in a_black])
        if y_a_min > y_a_max:
            continue
        
        # Calculate possible x_b range for B
        x_b_min = max([cx - i for i, j in b_black])
        x_b_max = min([(cx + h_x - 1) - i for i, j in b_black])
        if x_b_min > x_b_max:
            continue
        
        y_b_min = max([cy - j for i, j in b_black])
        y_b_max = min([(cy + w_x - 1) - j for i, j in b_black])
        if y_b_min > y_b_max:
            continue
        
        # Iterate all possible x_a, y_a, x_b, y_b in their ranges
        for x_a in range(x_a_min, x_a_max + 1):
            for y_a in range(y_a_min, y_a_max + 1):
                for x_b in range(x_b_min, x_b_max + 1):
                    for y_b in range(y_b_min, y_b_max + 1):
                        # Create the grid
                        grid = [['.' for _ in range(w_x)] for _ in range(h_x)]
                        
                        # Add contributions from A
                        for ai, aj in a_black:
                            pos_x = x_a + ai
                            pos_y = y_a + aj
                            if (cx <= pos_x <= cx + h_x - 1 and
                                cy <= pos_y <= cy + w_x - 1):
                                i = pos_x - cx
                                j = pos_y - cy
                                if 0 <= i < h_x and 0 <= j < w_x:
                                    grid[i][j] = '#'
                        
                        # Add contributions from B
                        for bi, bj in b_black:
                            pos_x = x_b + bi
                            pos_y = y_b + bj
                            if (cx <= pos_x <= cx + h_x - 1 and
                                cy <= pos_y <= cy + w_x - 1):
                                i = pos_x - cx
                                j = pos_y - cy
                                if 0 <= i < h_x and 0 <= j < w_x:
                                    grid[i][j] = '#'
                        
                        # Convert grid to list of strings
                        grid_str = [''.join(row) for row in grid]
                        
                        # Check if it matches X
                        if grid_str == x:
                            print("Yes")
                            exit()

# If no combination found
print("No")