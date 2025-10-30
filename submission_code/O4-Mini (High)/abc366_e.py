import sys
from array import array

def main():
    input = sys.stdin.readline
    N, D = map(int, input().split())
    xs = [0] * N
    ys = [0] * N
    for i in range(N):
        x, y = map(int, input().split())
        xs[i] = x
        ys[i] = y
    xs.sort()
    ys.sort()

    sum_xs = sum(xs)
    sum_ys = sum(ys)
    Xl = xs[0] - D
    Xr = xs[-1] + D
    Yl = ys[0] - D
    Yr = ys[-1] + D

    # Build count array C for G(y) values up to D
    C = array('I', [0]) * (D + 1)

    # Compute distribution of G(y) = sum_i |y - y_i|
    N_loc = N
    D_loc = D
    ys_loc = ys

    idx_y = 0
    pos_le_y = 0
    y_cur = Yl
    # initialize pos_le_y = # of y_i <= y_cur
    while idx_y < N_loc and ys_loc[idx_y] <= y_cur:
        pos_le_y += 1
        idx_y += 1
    # initial G(y_cur) since y_cur <= all ys_loc => G = sum_ys - N * y_cur
    G_cur = sum_ys - N_loc * y_cur

    Ly = Yr - Yl + 1
    C_arr = C
    for _ in range(Ly):
        if G_cur <= D_loc:
            C_arr[G_cur] += 1
        # update to G(y_cur+1)
        slope = (pos_le_y << 1) - N_loc
        G_cur += slope
        y_cur += 1
        # update pos_le_y for new y_cur
        while idx_y < N_loc and ys_loc[idx_y] <= y_cur:
            pos_le_y += 1
            idx_y += 1

    # prefix-sum so that C_arr[z] = # of y with G(y) <= z
    for i in range(1, D_loc + 1):
        C_arr[i] += C_arr[i - 1]

    # Now iterate over x and sum up valid pairs
    idx_x = 0
    pos_le_x = 0
    x_cur = Xl
    xs_loc = xs
    # initialize pos_le_x = # of x_i <= x_cur
    while idx_x < N_loc and xs_loc[idx_x] <= x_cur:
        pos_le_x += 1
        idx_x += 1
    # initial F(x_cur) since x_cur <= all xs_loc => F = sum_xs - N * x_cur
    F_cur = sum_xs - N_loc * x_cur

    ans = 0
    Lx = Xr - Xl + 1
    for _ in range(Lx):
        if F_cur <= D_loc:
            # add # of y with G(y) <= D - F_cur
            ans += C_arr[D_loc - F_cur]
        # update to F(x_cur+1)
        slope = (pos_le_x << 1) - N_loc
        F_cur += slope
        x_cur += 1
        # update pos_le_x for new x_cur
        while idx_x < N_loc and xs_loc[idx_x] <= x_cur:
            pos_le_x += 1
            idx_x += 1

    print(ans)

# call main to execute
main()