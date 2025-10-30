def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    grid = data[1:]
    
    count = 0
    
    # Precompute row and column counts of 'o'
    row_o_counts = [0] * N
    col_o_counts = [0] * N
    
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                row_o_counts[i] += 1
                col_o_counts[j] += 1
    
    # Find valid triples
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                # (i, j) is the common cell
                # We need at least one more 'o' in the same row and one more 'o' in the same column
                if row_o_counts[i] > 1 and col_o_counts[j] > 1:
                    # (row_o_counts[i] - 1) choices in row i, excluding (i, j)
                    # (col_o_counts[j] - 1) choices in column j, excluding (i, j)
                    count += (row_o_counts[i] - 1) * (col_o_counts[j] - 1)
    
    print(count)

if __name__ == "__main__":
    main()