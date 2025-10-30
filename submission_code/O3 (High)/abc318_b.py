import sys

def main() -> None:
    # Maximum possible coordinate value is 100, and all are integers.
    # We regard each unit square [x, x+1) × [y, y+1) (0 ≤ x, y ≤ 99) on the plane.
    # The union area equals the number of those unit squares covered by
    # at least one rectangle.
    MAX_COORD = 100                     # 0 … 100 -> 100 unit intervals

    grid = [[False] * MAX_COORD for _ in range(MAX_COORD)]  # grid[y][x]

    it = iter(sys.stdin.read().strip().split())
    N = int(next(it))

    for _ in range(N):
        A = int(next(it))
        B = int(next(it))
        C = int(next(it))
        D = int(next(it))

        # Mark all unit squares that lie inside the rectangle
        for y in range(C, D):           # y-coordinates C ≤ y < D
            row = grid[y]
            for x in range(A, B):       # x-coordinates A ≤ x < B
                row[x] = True

    # Count marked squares; each contributes area 1
    covered_area = sum(sum(row) for row in grid)
    print(covered_area)

if __name__ == "__main__":
    main()