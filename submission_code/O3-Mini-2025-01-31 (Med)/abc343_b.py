def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # First element is N
    N = int(input_data[0])
    # The rest elements are the matrix elements
    matrix_values = list(map(int, input_data[1:]))
    # Process each row
    # Each row will have N values sequentially in matrix_values list.
    idx = 0
    output_lines = []
    for i in range(N):
        # For vertex i+1, determine neighbors
        neighbors = []
        for j in range(N):
            if matrix_values[idx + j] == 1:
                # vertices are 1-indexed
                neighbors.append(str(j + 1))
        idx += N
        # prepare line: if neighbors are empty, it will be an empty string.
        output_lines.append(" ".join(neighbors))
    sys.stdout.write("
".join(output_lines))
    
if __name__ == "__main__":
    main()