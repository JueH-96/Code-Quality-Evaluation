from collections import defaultdict

# Read the grid dimensions and number of operations
H, W, M = map(int, input().split())

# Initialize the grid with color 0
grid = defaultdict(int)

# Keep track of the latest color for each row and column
latest_row_color = defaultdict(int)
latest_col_color = defaultdict(int)

# Keep track of the number of cells painted with each color
color_count = defaultdict(int)

# Initially, all cells are color 0
color_count[0] = H * W

# Process the operations
for _ in range(M):
    T, A, X = map(int, input().split())
    if T == 1:
        # Repaint row A with color X
        if A in latest_row_color:
            # Subtract the previous color count for this row
            color_count[latest_row_color[A]] -= W
        else:
            # Subtract the initial color 0 for this row
            color_count[0] -= W
        # Add the new color count for this row
        color_count[X] += W
        # Update the latest color for this row
        latest_row_color[A] = X
    else:
        # Repaint column A with color X
        if A in latest_col_color:
            # Subtract the previous color count for this column
            color_count[latest_col_color[A]] -= H
        else:
            # Subtract the initial color 0 for this column
            color_count[0] -= H
        # Add the new color count for this column
        color_count[X] += H
        # Update the latest color for this column
        latest_col_color[A] = X

    # Adjust for the intersection of the latest row and column colors
    for row, row_color in latest_row_color.items():
        for col, col_color in latest_col_color.items():
            if row_color == col_color:
                color_count[row_color] -= 1
            else:
                color_count[row_color] += 1
                color_count[col_color] += 1
                color_count[0] -= 1

# Remove colors with zero count
color_count = {color: count for color, count in color_count.items() if count > 0}

# Sort the colors
sorted_colors = sorted(color_count.keys())

# Output the number of distinct colors
print(len(sorted_colors))

# Output the color and the number of cells for each color
for color in sorted_colors:
    print(color, color_count[color])