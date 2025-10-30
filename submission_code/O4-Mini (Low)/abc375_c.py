def main():
    import sys
    input = sys.stdin.readline
    N = int(input())
    A = [input().rstrip('
') for _ in range(N)]
    # We'll compute for each cell how many nested-square rotations it undergoes:
    # layer index (0-based) = min(i, j, N-1-i, N-1-j)
    # number of 90° CW rotations = (layer + 1) % 4
    #
    # All these rotations are around the same center, so they accumulate into
    # a single rotation by total = ((layer+1)%4)*90°.
    # We perform the inverse map: for each original (i,j), compute its final
    # position (i2,j2), and assign B[i2][j2] = A[i][j].
    #
    # To avoid floats, do the rotation on coordinates multiplied by 2.
    C2 = N - 1  # 2 * center = N-1
    B = [list('.'*N) for _ in range(N)]
    for i in range(N):
        for j in range(N):
            # determine layer
            layer = min(i, j, N-1-i, N-1-j)
            rot = (layer + 1) & 3  # number of 90° CW rotations mod 4
            # coordinates relative to center, scaled by 2
            dx2 = 2*i - C2
            dy2 = 2*j - C2
            # apply rot times: (dx2,dy2) -> (-dy2, dx2)
            # do it in a small loop since rot <= 3
            for _ in range(rot):
                dx2, dy2 = -dy2, dx2
            # back to grid coords
            i2 = (dx2 + C2) >> 1
            j2 = (dy2 + C2) >> 1
            B[i2][j2] = A[i][j]
    # output
    out = sys.stdout
    for row in B:
        out.write(''.join(row) + '
')

if __name__ == "__main__":
    main()