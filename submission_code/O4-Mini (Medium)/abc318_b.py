def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    # We only need to consider unit squares [i, i+1) x [j, j+1) for i,j in [0..99].
    # Create a 100x100 grid to mark covered cells.
    covered = [[False] * 100 for _ in range(100)]

    for _ in range(N):
        A, B, C, D = map(int, input().split())
        # Mark all integer grid cells (i,j) where A <= i < B and C <= j < D
        for i in range(A, B):
            for j in range(C, D):
                covered[i][j] = True

    # Count how many unit cells are covered
    area = 0
    for i in range(100):
        for j in range(100):
            if covered[i][j]:
                area += 1

    print(area)

if __name__ == "__main__":
    main()