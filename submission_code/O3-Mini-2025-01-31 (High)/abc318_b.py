def main():
    import sys

    data = sys.stdin.read().strip().split()
    if not data:
        return

    it = iter(data)
    N = int(next(it))
    # Since the maximum coordinates are 100 (B_i, D_i ≤ 100) and the rectangle covers
    # region [A, B] x [C, D] correspond to unit square areas (area = (B-A)*(D-C)),
    # we can use a 2D grid of size 101 x 101 and mark each cell that is covered.
    grid = [[False] * 101 for _ in range(101)]
    
    for _ in range(N):
        A = int(next(it))
        B = int(next(it))
        C = int(next(it))
        D = int(next(it))
        # Mark cells corresponding to each sheet.
        # We consider the grid cell's lower left coordinate (x, y) where
        # the covering cell is from (x,y) to (x+1, y+1).
        # Since the area of the rectangle is (B-A) * (D-C), we mark x in [A, B) and y in [C, D).
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = True

    # Calculate the total union area as the total count of covered cells.
    area = sum(sum(1 for cell in row if cell) for row in grid)
    print(area)

if __name__ == '__main__':
    main()