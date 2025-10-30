import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, D = map(int, input().split())
    xs = [0]*N
    ys = [0]*N
    for i in range(N):
        x,y = map(int, input().split())
        xs[i], ys[i] = x,y
    xs.sort()
    ys.sort()

    # Domain for x and y
    min_x = xs[0]
    max_x = xs[-1]
    Lx = min_x - D
    Rx = max_x + D
    min_y = ys[0]
    max_y = ys[-1]
    Ly = min_y - D
    Ry = max_y + D
    Wx = Rx - Lx + 1
    Wy = Ry - Ly + 1

    # medians
    med_x = xs[N//2]
    med_y = ys[N//2]

    # Compute A0 = sum_i |med_x - xs[i]|, and slope Sx at med_x going right:
    # slope_right(x) = #xi ≤ x  − (#xi > x) = 2*cnt_le - N
    import bisect
    cnt_le_x = bisect.bisect_right(xs, med_x)
    A0 = 0
    for v in xs:
        A0 += abs(v - med_x)
    Sx0 = 2*cnt_le_x - N

    # Build Bval[y] = sum_i |y - ys[i]| for y in [Ly..Ry]
    # We do a 1D sweep from y = Ly..Ry, tracking slope_by:
    # slope_by(y) = #yi ≤ y  − (#yi > y) similarly.
    Bval = [0]*Wy
    # initial at y=Ly
    # sum_i |Ly - ys[i]| = sum(ys) - N*Ly
    sum_ys = sum(ys)
    B0 = sum_ys - N*Ly
    Bval[0] = B0
    cnt_le = bisect.bisect_right(ys, Ly)
    slope = 2*cnt_le - N
    p = cnt_le
    # sweep
    for j in range(1, Wy):
        # y = Ly + j
        B0 += slope
        y_now = Ly + j
        # any ys[p] == y_now?
        while p < N and ys[p] == y_now:
            slope += 2
            p += 1
        Bval[j] = B0

    # locate index of med_y in Bval
    y0_idx = med_y - Ly
    # sliding window in y
    ans = 0

    # initial x = med_x
    A = A0
    Sx = Sx0
    # initial window for B[y] ≤ D - A
    T = D - A
    # l..r inclusive
    if 0 <= y0_idx < Wy and Bval[y0_idx] <= T:
        l = y0_idx
        while l > 0 and Bval[l-1] <= T:
            l -= 1
        r = y0_idx
        while r+1 < Wy and Bval[r+1] <= T:
            r += 1
        ans += (r - l + 1)
    else:
        # no y works
        l = y0_idx+1
        r = y0_idx-1

    # move x from med_x+1 .. Rx
    # maintain cnt_le_x and slope Sx
    p = cnt_le_x
    x = med_x
    for x in range(med_x+1, Rx+1):
        A += Sx
        # update slope when crossing xi == x
        while p < N and xs[p] == x:
            Sx += 2
            p += 1
        T = D - A
        # shrink window if needed
        # l moves up if Bval[l] > T
        while l <= y0_idx and l < Wy and Bval[l] > T:
            l += 1
        # r moves down if Bval[r] > T
        while r >= y0_idx and r >= 0 and Bval[r] > T:
            r -= 1
        if l <= r:
            ans += (r - l + 1)

    # move x from med_x-1 .. Lx
    # for left, slope_left(x) = #xi ≥ x  − (#xi < x) = N - 2*cnt_less
    # initial cnt_less at med_x = #xi < med_x
    cnt_less = bisect.bisect_left(xs, med_x)
    Sleft = N - 2*cnt_less
    A = A0
    l2, r2 = l, r  # continue adjusting same window
    x = med_x
    for x in range(med_x-1, Lx-1, -1):
        A += Sleft
        # update when crossing xi == x
        # those xi which were counted in "< x+1" but not in "< x"
        while cnt_less > 0 and xs[cnt_less-1] == x:
            cnt_less -= 1
            Sleft += 2
        T = D - A
        while l2 <= y0_idx and l2 < Wy and Bval[l2] > T:
            l2 += 1
        while r2 >= y0_idx and r2 >= 0 and Bval[r2] > T:
            r2 -= 1
        if l2 <= r2:
            ans += (r2 - l2 + 1)

    print(ans)

if __name__ == "__main__":
    main()