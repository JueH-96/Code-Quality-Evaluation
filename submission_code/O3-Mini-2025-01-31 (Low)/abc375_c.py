def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    grid = [list(input().strip()) for _ in range(N)]
    
    # Process each layer k from outer to inner.
    for k in range(N // 2):
        # Dimension of the current subgrid.
        n = N - 2 * k
        
        # Create a copy (temporary grid) for the subgrid from rows k to N-1-k, columns k to N-1-k.
        temp = [grid[k + i][k: k + n] for i in range(n)]
        
        # Apply the transformation: for each cell in the subgrid, assign rotated value.
        for i in range(n):
            for j in range(n):
                # Mapping: (i, j) in temp goes to (j, n-1-i) in the subgrid of grid.
                grid[k + j][N - 1 - k - i] = temp[i][j]
    
    # Output the final grid.
    out_lines = ["".join(row) for row in grid]
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()