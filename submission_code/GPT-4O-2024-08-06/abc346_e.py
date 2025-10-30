# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    M = int(data[2])
    
    operations = []
    index = 3
    for _ in range(M):
        T = int(data[index])
        A = int(data[index + 1])
        X = int(data[index + 2])
        operations.append((T, A, X))
        index += 3
    
    # Track the last color applied to each row and column
    last_row_color = [-1] * H
    last_col_color = [-1] * W
    
    # Track the last operation index for each row and column
    last_row_op = [-1] * H
    last_col_op = [-1] * W
    
    # Process operations to determine the final color of each row and column
    for op_index, (T, A, X) in enumerate(operations):
        if T == 1:
            last_row_color[A - 1] = X
            last_row_op[A - 1] = op_index
        elif T == 2:
            last_col_color[A - 1] = X
            last_col_op[A - 1] = op_index
    
    # Count colors
    color_count = {}
    
    # Determine the final color of each cell
    for r in range(H):
        for c in range(W):
            # Determine which operation was the last for this cell
            if last_row_op[r] > last_col_op[c]:
                # Row operation was last
                final_color = last_row_color[r]
            else:
                # Column operation was last
                final_color = last_col_color[c]
            
            # Count the color
            if final_color in color_count:
                color_count[final_color] += 1
            else:
                color_count[final_color] = 1
    
    # Prepare output
    distinct_colors = sorted(color_count.keys())
    print(len(distinct_colors))
    for color in distinct_colors:
        print(color, color_count[color])

if __name__ == "__main__":
    main()