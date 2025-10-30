def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    # read N
    N = int(data[0])
    index = 1
    
    # Build matrix: A[i][j] where i, j are 1-indexed.
    # But we'll store in a 2D list for i from 0 to N-1 and j from 0 to i.
    # Actually, input gives row i with i numbers.
    matrix = []
    for i in range(N):
        row = []
        for j in range(i+1):
            row.append(int(data[index]))
            index += 1
        matrix.append(row)
    
    # Start with element 1.
    element = 1
    for nxt in range(1, N+1):
        if element >= nxt:
            # use A_{element, nxt} -> matrix[element-1][nxt-1] because element and nxt are 1-indexed.
            element = matrix[element-1][nxt-1]
        else:
            # element < nxt -> use A_{nxt, element}
            element = matrix[nxt-1][element-1]
    sys.stdout.write(str(element))
    
if __name__ == '__main__':
    main()