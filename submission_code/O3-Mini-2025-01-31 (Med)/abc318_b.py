def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    
    # The maximum coordinate is 100.
    # Each rectangle covers [A, B] x [C, D], and its area is (B-A)*(D-C)
    # We can treat the plane as composed of 1x1 unit squares.
    # A sheet that covers [A, B] x [C, D] will cover each unit square with its bottom‐left corner (x, y)
    # for x in range(A, B) and y in range(C, D).
    grid = [[False] * 100 for _ in range(100)]
    
    for _ in range(N):
        A = int(next(it))
        B = int(next(it))
        C = int(next(it))
        D = int(next(it))
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = True
                
    total_area = sum(sum(row) for row in grid)
    sys.stdout.write(str(total_area))

if __name__ == '__main__':
    main()