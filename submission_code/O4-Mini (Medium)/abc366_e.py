import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N = int(line[0]); D = int(line[1])
    xs = [0]*N
    ys = [0]*N
    for i in range(N):
        xi, yi = map(int, data.readline().split())
        xs[i] = xi
        ys[i] = yi
    xs.sort()
    ys.sort()
    # prefix sums for xs
    Sx = [0] * (N+1)
    for i in range(N):
        Sx[i+1] = Sx[i] + xs[i]
    Sx_total = Sx[N]
    # function to compute A(x)
    import bisect
    def A_val(x):
        k = bisect.bisect_right(xs, x)
        sum_left = Sx[k]
        sum_right = Sx_total - sum_left
        # sum |x - xi| = x*k - sum_left + sum_right - x*(N-k)
        return x*k - sum_left + sum_right - x*(N-k)

    # find median x
    xm = xs[N//2]
    # if minimal A > D, no points
    if A_val(xm) > D:
        print(0)
        return
    # binary search leftmost xL in [xm-D, xm] s.t. A(xL) <= D
    lo = xm - D
    hi = xm
    # adjust lo, hi inclusive
    while lo < hi:
        mid = (lo + hi) // 2
        if A_val(mid) <= D:
            hi = mid
        else:
            lo = mid + 1
    xL = lo
    # binary search rightmost xR in [xm, xm+D]
    lo = xm
    hi = xm + D
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if A_val(mid) <= D:
            lo = mid
        else:
            hi = mid - 1
    xR = lo

    # Prepare C[rem] = number of integer y s.t. sum_i |y - yi| <= rem
    # Build cost_left and cost_right up to rem<=D
    from collections import Counter
    cty = Counter(ys)
    # median y
    ym = ys[N//2]
    # k_left = count of yi <= ym
    # to get that, sum counts for y <= ym
    # faster: use bisect on ys
    k_left = bisect.bisect_right(ys, ym)
    # cost_left: costs for offsets d>=0: cost_left[0]=0, cost at y=ym-d
    cost_left = [0]
    # count of yi < ym:
    cnt_eq = cty.get(ym, 0)
    curr_k_le = k_left - cnt_eq  # count yi < ym
    curr_cost = 0
    y_curr = ym
    # build left until cost > D
    while True:
        # delta going from y_curr to y_curr-1
        # delta = B(y_curr-1)-B(y_curr) = N - 2*(# yi <= y_curr-1)
        delta = N - 2 * curr_k_le
        curr_cost += delta
        if curr_cost > D:
            break
        cost_left.append(curr_cost)
        # move to next
        y_curr -= 1
        curr_k_le -= cty.get(y_curr, 0)
    # cost_right: for y = ym + d
    cost_right = [0]
    curr_cost = 0
    curr_k_le = k_left  # count yi <= ym
    y_curr = ym
    while True:
        # delta = B(y_curr+1)-B(y_curr) = (# yi <= y_curr) - (# yi > y_curr)
        delta = curr_k_le - (N - curr_k_le)
        curr_cost += delta
        if curr_cost > D:
            break
        cost_right.append(curr_cost)
        # move
        y_curr += 1
        curr_k_le += cty.get(y_curr, 0)

    # build C array
    C = [0] * (D+1)
    pL = 0
    pR = 0
    lenL = len(cost_left)
    lenR = len(cost_right)
    for rem in range(D+1):
        # advance pL
        while pL+1 < lenL and cost_left[pL+1] <= rem:
            pL += 1
        # advance pR
        while pR+1 < lenR and cost_right[pR+1] <= rem:
            pR += 1
        # total ys = (pL + pR + 1)
        C[rem] = pL + pR + 1

    # Now sweep x from xL to xR, maintain A, k_x, slope
    ans = 0
    # initial A and k_x
    A = A_val(xL)
    # count xi <= xL via bisect_right
    kx = bisect.bisect_right(xs, xL)
    # slope = A(x+1)-A(x) = 2*kx - N
    slope = 2 * kx - N
    # pointer in xs for updating kx
    p_x = kx
    x = xL
    # iterate
    # for x in [xL..xR]
    for _ in range(xR - xL + 1):
        rem = D - A
        if rem >= 0:
            ans += C[rem]
        # move to next x
        A += slope
        x += 1
        # update kx with xs equal to new x
        while p_x < N and xs[p_x] == x:
            kx += 1
            p_x += 1
        slope = 2 * kx - N

    print(ans)


if __name__ == "__main__":
    main()