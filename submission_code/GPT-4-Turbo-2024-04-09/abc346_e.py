import sys
input = sys.stdin.read

def main():
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
    
    # Latest color for each row and column
    row_colors = [-1] * (H + 1)
    col_colors = [-1] * (W + 1)
    
    # Latest operation index that affected each row and column
    row_op_index = [-1] * (H + 1)
    col_op_index = [-1] * (W + 1)
    
    # Process operations
    for op_index, (T, A, X) in enumerate(operations):
        if T == 1:
            row_colors[A] = X
            row_op_index[A] = op_index
        elif T == 2:
            col_colors[A] = X
            col_op_index[A] = op_index
    
    # Final grid color count
    color_count = {}
    
    # Determine the final color of each cell
    for r in range(1, H + 1):
        for c in range(1, W + 1):
            # Determine the effective color by the latest operation
            if row_op_index[r] > col_op_index[c]:
                color = row_colors[r]
            else:
                color = col_colors[c]
            
            if color in color_count:
                color_count[color] += 1
            else:
                color_count[color] = 1
    
    # Output results
    colors = sorted(color_count.items())
    print(len(colors))
    for color, count in colors:
        print(color, count)

if __name__ == "__main__":
    main()