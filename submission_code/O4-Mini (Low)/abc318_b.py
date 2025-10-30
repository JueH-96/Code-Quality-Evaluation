def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    # Since coordinates go from 0 to 100 and are integers, we can discretize by unit squares.
    # grid[x][y] = True if the unit cell [x, x+1) x [y, y+1) is covered.
    grid = [[False]*100 for _ in range(100)]

    for _ in range(N):
        A, B, C, D = map(int, input().split())
        # Mark all unit cells covered by this rectangle
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = True

    # Count covered cells
    area = 0
    for x in range(100):
        # Using sum on booleans: True == 1, False == 0
        area += sum(grid[x])

    print(area)

if __name__ == "__main__":
    main()