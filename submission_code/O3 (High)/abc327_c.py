def main():
    import sys

    # Read the 9×9 grid
    A = [list(map(int, sys.stdin.readline().split())) for _ in range(9)]

    full = set(range(1, 10))          # The set {1,2,…,9}

    # Check every row
    for row in A:
        if set(row) != full:
            print("No")
            return

    # Check every column
    for c in range(9):
        col = {A[r][c] for r in range(9)}
        if col != full:
            print("No")
            return

    # Check every 3×3 sub-grid
    for sr in (0, 3, 6):          # sub-grid starting row
        for sc in (0, 3, 6):      # sub-grid starting column
            block = set()
            for r in range(sr, sr + 3):
                for c in range(sc, sc + 3):
                    block.add(A[r][c])
            if block != full:
                print("No")
                return

    # All conditions satisfied
    print("Yes")


if __name__ == "__main__":
    main()