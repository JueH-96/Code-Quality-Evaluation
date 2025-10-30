def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    # Build the triangular matrix: A[i][j] will represent A_{i+1, j+1} for j<=i.
    matrix = []
    index = 1
    for i in range(1, N + 1):
        row = []
        for j in range(i):
            row.append(int(data[index]))
            index += 1
        matrix.append(row)
    
    # Start with element 1.
    current = 1
    # For each element 1, 2, ..., N combine in order.
    for x in range(1, N + 1):
        # If current >= x, use A_{current, x} = matrix[current-1][x-1]
        if current >= x:
            current = matrix[current - 1][x - 1]
        else:
            # Otherwise use A_{x, current} = matrix[x-1][current-1]
            current = matrix[x - 1][current - 1]
    sys.stdout.write(str(current))

if __name__ == '__main__':
    main()