def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    # Parse grid dimensions
    H, W = map(int, input_data[0].split())
    grid = [list(line) for line in input_data[1:1+H]]
    
    # Find the bounding rectangle of the already black cells ('#')
    rmin, rmax = H, -1
    cmin, cmax = W, -1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                rmin = min(rmin, i)
                rmax = max(rmax, i)
                cmin = min(cmin, j)
                cmax = max(cmax, j)
    
    # Now, within the bounding rectangle, all cells must end up black.
    # Any cell already painted white ('.') in this region makes it impossible.
    for i in range(rmin, rmax + 1):
        for j in range(cmin, cmax + 1):
            if grid[i][j] == '.':
                print("No")
                return
    
    # Outside the rectangle, no cell is forced to be black (we can always paint them white).
    # Hence, it's possible to complete the painting.
    print("Yes")
    
if __name__ == '__main__':
    main()