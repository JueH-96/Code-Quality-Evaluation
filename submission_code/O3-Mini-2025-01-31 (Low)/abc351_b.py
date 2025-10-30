def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    n = int(input_data[0])
    # Read grid A lines
    gridA = input_data[1:n+1]
    # Read grid B lines
    gridB = input_data[n+1: 2*n+1]
    
    # Loop through each cell and compare, there is exactly one difference.
    for i in range(n):
        for j in range(n):
            if gridA[i][j] != gridB[i][j]:
                # Since the problem expects 1-indexed, adjust output.
                print(i + 1, j + 1)
                return

if __name__ == '__main__':
    main()