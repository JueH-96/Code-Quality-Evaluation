def main():
    import sys
    input = sys.stdin.readline
    
    H, W, N = map(int, input().split())
    T = input().strip()
    grid = [input().strip() for _ in range(H)]
    
    # Build land bitmasks: for each row, a W-bit integer with 1 where cell is land ('.')
    land = [0] * H
    for i in range(H):
        mask = 0
        row = grid[i]
        # we map column j in [0..W-1] to bit (1 << j)
        for j, c in enumerate(row):
            if c == '.':
                mask |= 1 << j
        land[i] = mask
    
    # possible positions at current step: init to all land cells
    possible = land.copy()
    
    # mask to truncate shifts to W bits
    full_mask = (1 << W) - 1
    
    for mv in T:
        newp = [0] * H
        if mv == 'L':
            # move left: from (i, j+1) to (i, j)
            for i in range(H):
                # shift right by 1 and AND with land
                newp[i] = (possible[i] >> 1) & land[i]
        elif mv == 'R':
            # move right: from (i, j-1) to (i, j)
            for i in range(H):
                newp[i] = ((possible[i] << 1) & full_mask) & land[i]
        elif mv == 'U':
            # move up: from (i+1, j) to (i, j)
            # last row has no above
            for i in range(H - 1):
                newp[i] = possible[i+1] & land[i]
            # newp[H-1] stays 0
        elif mv == 'D':
            # move down: from (i-1, j) to (i, j)
            for i in range(1, H):
                newp[i] = possible[i-1] & land[i]
            # newp[0] stays 0
        else:
            # should not happen
            pass
        
        possible = newp
    
    # count total bits in possible
    ans = 0
    for row_mask in possible:
        ans += row_mask.bit_count()
    
    print(ans)

if __name__ == "__main__":
    main()