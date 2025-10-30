def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # First token is N, then N tokens for grid A rows and the next N tokens for grid B rows.
    N = int(input_data[0])
    gridA = input_data[1:1+N]
    gridB = input_data[1+N:1+2*N]

    # Iterate over each cell in the grids.
    for i in range(N):
        for j in range(N):
            if gridA[i][j] != gridB[i][j]:
                # We output using 1-indexed positions.
                print(i + 1, j + 1)
                return

# Do not forget to call main()
if __name__ == '__main__':
    main()