def can_color_grid(N, M, colored_cells):
    # Initialize the limits for rows and columns
    row_limit = [N] * N  # Maximum black cells allowed in each row
    col_limit = [N] * N  # Maximum black cells allowed in each column

    # Process the colored cells
    for x, y, color in colored_cells:
        row_index = x - 1  # Convert to 0-based index
        col_index = y - 1  # Convert to 0-based index
        
        if color == 'B':
            # If it's black, we cannot have more black cells to the right
            row_limit[row_index] = min(row_limit[row_index], col_index)
            # If it's black, we cannot have more black cells below
            col_limit[col_index] = min(col_limit[col_index], row_index)
        elif color == 'W':
            # If it's white, we cannot have black cells to the left
            row_limit[row_index] = min(row_limit[row_index], col_index + 1)
            # If it's white, we cannot have black cells above
            col_limit[col_index] = min(col_limit[col_index], row_index + 1)

    # Check if the limits are consistent
    for i in range(N):
        if row_limit[i] < 0 or col_limit[i] < 0:
            return "No"
    
    # Check if the row limits and column limits are consistent
    for i in range(N):
        if row_limit[i] < col_limit[i]:
            return "No"

    return "Yes"

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N, M = map(int, data[0].split())
    colored_cells = []
    
    for i in range(1, M + 1):
        x, y, c = data[i].split()
        colored_cells.append((int(x), int(y), c))
    
    result = can_color_grid(N, M, colored_cells)
    print(result)

if __name__ == "__main__":
    main()