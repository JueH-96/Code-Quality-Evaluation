def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    
    # Create a grid of size 101x101 to cover coordinates [0, 100).
    # Each cell at (x, y) corresponds to the unit square between x and x+1 and between y and y+1.
    grid = [[False] * 101 for _ in range(101)]
    
    # For each sheet, mark the covered cells.
    for _ in range(N):
        A = int(next(it))
        B = int(next(it))
        C = int(next(it))
        D = int(next(it))
        
        # Since the rectangle covers the region A <= x <= B and C <= y <= D,
        # its area is calculated as (B - A) * (D - C).
        # Here, for each integer x in [A, B-1] and y in [C, D-1], mark the corresponding unit square.
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = True

    # Sum the total area covered by one or more sheets.
    area = sum(cell for row in grid for cell in row)
    sys.stdout.write(str(area))
    
if __name__ == '__main__':
    main()