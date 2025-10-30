def main():
    import sys

    # Read the 9x9 grid
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(9)]
    
    # Precomputed valid set for 1..9
    valid = set(range(1, 10))
    
    # Check each row
    for row in A:
        if set(row) != valid:
            print("No")
            return
    
    # Check each column
    for c in range(9):
        col = {A[r][c] for r in range(9)}
        if col != valid:
            print("No")
            return
    
    # Check each 3x3 subgrid
    for br in (0, 3, 6):
        for bc in (0, 3, 6):
            block = set()
            for dr in range(3):
                for dc in range(3):
                    block.add(A[br + dr][bc + dc])
            if block != valid:
                print("No")
                return
    
    # If all checks pass
    print("Yes")

if __name__ == "__main__":
    main()