def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # Convert input into a 9x9 grid of integers
    A = []
    for i in range(9):
        A.append(list(map(int, data[i*9:(i+1)*9])))
    
    # Check rows
    for row in A:
        if sorted(row) != list(range(1, 10)):
            print("No")
            return
    
    # Check columns
    for j in range(9):
        col = [A[i][j] for i in range(9)]
        if sorted(col) != list(range(1, 10)):
            print("No")
            return
    
    # Check 3x3 subgrids
    for start_row in range(0, 9, 3):
        for start_col in range(0, 9, 3):
            subgrid = []
            for i in range(3):
                for j in range(3):
                    subgrid.append(A[start_row + i][start_col + j])
            if sorted(subgrid) != list(range(1, 10)):
                print("No")
                return
    
    # If all checks pass
    print("Yes")

if __name__ == "__main__":
    main()