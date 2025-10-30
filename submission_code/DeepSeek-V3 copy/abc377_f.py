# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    rows = set()
    cols = set()
    diag1 = set()
    diag2 = set()
    
    for i in range(M):
        a = int(data[2 + 2*i])
        b = int(data[2 + 2*i + 1])
        rows.add(a)
        cols.add(b)
        diag1.add(a + b)
        diag2.add(a - b)
    
    total = N * N
    occupied = M
    unsafe = 0
    
    # Rows
    unsafe += len(rows) * N
    # Columns
    unsafe += len(cols) * N
    # Diagonals
    unsafe += len(diag1) * N
    unsafe += len(diag2) * N
    
    # Intersections
    # Rows and columns
    unsafe -= len(rows) * len(cols)
    # Rows and diag1
    unsafe -= len(rows) * len(diag1)
    # Rows and diag2
    unsafe -= len(rows) * len(diag2)
    # Columns and diag1
    unsafe -= len(cols) * len(diag1)
    # Columns and diag2
    unsafe -= len(cols) * len(diag2)
    # Diag1 and diag2
    unsafe -= len(diag1) * len(diag2)
    
    # Add back the intersections of three sets
    # Rows, columns, and diag1
    unsafe += len(rows) * len(cols) * len(diag1)
    # Rows, columns, and diag2
    unsafe += len(rows) * len(cols) * len(diag2)
    # Rows, diag1, and diag2
    unsafe += len(rows) * len(diag1) * len(diag2)
    # Columns, diag1, and diag2
    unsafe += len(cols) * len(diag1) * len(diag2)
    
    # Subtract the intersections of all four sets
    unsafe -= len(rows) * len(cols) * len(diag1) * len(diag2)
    
    # The total unsafe squares are the sum of the above, but we need to subtract the occupied squares
    # since they are already counted in the unsafe squares
    unsafe -= M
    
    # The safe squares are total squares minus unsafe squares
    safe = total - unsafe
    
    print(safe)

if __name__ == "__main__":
    main()