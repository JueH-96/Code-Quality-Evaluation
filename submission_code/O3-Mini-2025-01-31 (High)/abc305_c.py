def main():
    import sys
    input_data = sys.stdin.read().splitlines()
    if not input_data:
        return
    
    # Parse H and W
    H, W = map(int, input_data[0].split())
    grid = input_data[1:1+H]
    
    # Find the bounding box of all '#' cells.
    rmin, rmax = H, -1
    cmin, cmax = W, -1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                if i < rmin:
                    rmin = i
                if i > rmax:
                    rmax = i
                if j < cmin:
                    cmin = j
                if j > cmax:
                    cmax = j

    # The original rectangle had cookies in all cells from (rmin,cmin) to (rmax,cmax)
    # except one cell where Snuke ate a cookie.
    # Find that missing cell in the bounding rectangle.
    for i in range(rmin, rmax+1):
        for j in range(cmin, cmax+1):
            if grid[i][j] == '.':
                # Convert 0-indexed to 1-indexed and output.
                print(i+1, j+1)
                return

if __name__ == '__main__':
    main()