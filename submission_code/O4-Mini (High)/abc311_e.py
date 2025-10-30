import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        H = int(next(it))
    except StopIteration:
        return
    W = int(next(it))
    N = int(next(it))

    # holes[row][col] = 1 if there's a hole at (row,col)
    holes = [bytearray(W+1) for _ in range(H+1)]
    for _ in range(N):
        r = int(next(it)); c = int(next(it))
        holes[r][c] = 1

    # We only need two rows of DP at a time
    prev = [0] * (W+1)
    cur  = [0] * (W+1)
    total = 0

    for i in range(1, H+1):
        row_holes = holes[i]
        cur[0] = 0
        p = prev
        c = cur
        ti = total
        for j in range(1, W+1):
            if row_holes[j]:
                c[j] = 0
            else:
                aj = p[j]
                bj = c[j-1]
                dj = p[j-1]
                # compute m = min(aj, bj, dj) by explicit comparisons
                if aj < bj:
                    if aj < dj:
                        m = aj
                    else:
                        m = dj
                else:
                    if bj < dj:
                        m = bj
                    else:
                        m = dj
                v = m + 1
                c[j] = v
                ti += v
        total = ti
        prev, cur = c, p

    sys.stdout.write(str(total))

if __name__ == "__main__":
    main()