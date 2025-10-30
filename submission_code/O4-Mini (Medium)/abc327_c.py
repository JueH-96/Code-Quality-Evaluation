def main():
    import sys
    
    # Read the 9x9 grid
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(9)]
    
    required = set(range(1, 10))
    
    # 1) Check each row
    for row in A:
        if set(row) != required:
            print("No")
            return
    
    # 2) Check each column
    for c in range(9):
        col = {A[r][c] for r in range(9)}
        if col != required:
            print("No")
            return
    
    # 3) Check each 3x3 subgrid
    for bi in range(3):
        for bj in range(3):
            block = set()
            for di in range(3):
                for dj in range(3):
                    block.add(A[3*bi + di][3*bj + dj])
            if block != required:
                print("No")
                return
    
    # If all checks pass
    print("Yes")

if __name__ == "__main__":
    main()