def main():
    import sys
    data = sys.stdin.read().split()
    sx = int(data[0]); sy = int(data[1])
    tx = int(data[2]); ty = int(data[3])
    # find the "left" coordinate of the horizontal domino covering (sx,sy)
    # if (sx+sy) is odd, we subtract 1 to get the left cell; else leave as is
    sd = sx - (((sx & 1) ^ (sy & 1)))
    td = tx - (((tx & 1) ^ (ty & 1)))
    # differences
    dd = td - sd
    dy = ty - sy
    # compute a = dd+dy, b = dd-dy
    a = dd + dy
    b = dd - dy
    # the minimum number of tolls is
    # (max(a,b,0) - min(a,b,0)) // 2
    mx = a if a > b else b
    if 0 > mx:
        mx = 0
    mn = a if a < b else b
    if 0 < mn:
        mn = 0
    ans = (mx - mn) // 2
    print(ans)

if __name__ == "__main__":
    main()