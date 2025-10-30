def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B, C, D = map(int, data)

    # helper for A1(w): doubled black area for vertical strip width w, height 2
    def A1(w):
        # w is integer 1 or 2
        if w <= 0:
            return 0
        if w == 1:
            return 3
        # w==2
        return 6

    # helper for A2(h): doubled black area for horizontal strip width 2, height h
    def A2(h):
        # h is integer 1 or 2
        if h <= 0:
            return 0
        if h == 1:
            return 3
        # h==2
        return 6

    # helper for A3(w,h): doubled black area for small corner w<2,h<2
    def A3(w, h):
        # only w==1 and h==1 occurs
        if w <= 0 or h <= 0:
            return 0
        # w==1,h==1
        return 2

    # Compute block-index ranges for x
    il = A // 2
    iu = (C - 1) // 2
    # Compute intersection width for leftmost block i=il
    # block span in x is [2*il, 2*il+2)
    w_left = 0
    if il <= iu:
        start = A - 2*il
        if start < 0:
            start = 0
        end = C - 2*il
        if end > 2:
            end = 2
        if end > start:
            w_left = end - start
    # for rightmost block i=iu (if distinct from il)
    w_right = 0
    if iu > il:
        start = A - 2*iu
        if start < 0:
            start = 0
        end = C - 2*iu
        if end > 2:
            end = 2
        if end > start:
            w_right = end - start
    # count of middle full-width blocks
    total_x_blocks = max(0, iu - il + 1)
    cnt_edge = 0
    if w_left > 0:
        cnt_edge += 1
    if w_right > 0:
        cnt_edge += 1
    n_mid_x = total_x_blocks - cnt_edge
    if n_mid_x < 0:
        n_mid_x = 0

    # Similarly for y
    jl = B // 2
    ju = (D - 1) // 2
    h_bot = 0
    if jl <= ju:
        start = B - 2*jl
        if start < 0:
            start = 0
        end = D - 2*jl
        if end > 2:
            end = 2
        if end > start:
            h_bot = end - start
    h_top = 0
    if ju > jl:
        start = B - 2*ju
        if start < 0:
            start = 0
        end = D - 2*ju
        if end > 2:
            end = 2
        if end > start:
            h_top = end - start
    total_y_blocks = max(0, ju - jl + 1)
    cnt_edge_y = 0
    if h_bot > 0:
        cnt_edge_y += 1
    if h_top > 0:
        cnt_edge_y += 1
    n_mid_y = total_y_blocks - cnt_edge_y
    if n_mid_y < 0:
        n_mid_y = 0

    # Build arrays for cw=0(left),1(mid),2(right)
    width = [0, 0, 0]
    cnt_i = [0, 0, 0]
    cnt_i_even = [0, 0, 0]
    cnt_i_odd = [0, 0, 0]

    # left column
    if w_left > 0:
        width[0] = w_left
        cnt_i[0] = 1
        if (il & 1) == 0:
            cnt_i_even[0] = 1
        else:
            cnt_i_odd[0] = 1
    # mid columns
    if n_mid_x > 0:
        # all these blocks have full width 2
        width[1] = 2
        cnt_i[1] = n_mid_x
        # count how many of these i are even
        # i runs from il+1 to iu-1 inclusive
        l = il + 1
        r = iu - 1
        if r >= l:
            # number of evens in [l,r]
            # even count = floor(r/2) - floor((l-1)/2)
            cnt_ev = (r // 2) - ((l - 1) // 2)
            if cnt_ev < 0:
                cnt_ev = 0
            if cnt_ev > n_mid_x:
                cnt_ev = n_mid_x
            cnt_i_even[1] = cnt_ev
            cnt_i_odd[1] = n_mid_x - cnt_ev
        else:
            # no mid
            cnt_i_even[1] = 0
            cnt_i_odd[1] = 0
    # right column
    if w_right > 0:
        width[2] = w_right
        cnt_i[2] = 1
        if (iu & 1) == 0:
            cnt_i_even[2] = 1
        else:
            cnt_i_odd[2] = 1

    # Similarly for rows ch=0(bot),1(mid),2(top)
    height = [0, 0, 0]
    cnt_j = [0, 0, 0]
    # j parity doesn't matter for flipping, only i parity does

    # bottom row
    if h_bot > 0:
        height[0] = h_bot
        # treat as partial only if height<2
        if h_bot < 2:
            cnt_j[0] = 1
        else:
            # full height row
            height[1] = 2  # include in mid
        # if exactly 2, it's a full row
    # top row
    if h_top > 0:
        height[2] = h_top
        if ju > jl:  # only if distinct
            if h_top < 2:
                cnt_j[2] = 1
            else:
                # full row => mid
                height[1] = 2
    # now count mid rows
    # total_y_blocks = cnt_j[0] + cnt_j[2] + n_mid_y
    cnt_j[1] = n_mid_y
    # if bottom or top had height==2 but were not counted as edge, they are part of mid
    # we handled that by not counting them in cnt_j[0] or cnt_j[2] if height==2

    # Now compute answer
    ans = 0
    # iterate over cw,ch
    for cw in range(3):
        w = width[cw]
        if w <= 0:
            continue
        for ch in range(3):
            h = height[ch]
            nj = cnt_j[ch]
            if nj == 0 or h <= 0:
                continue
            # determine f_even for block of size w x h
            if w == 2 and h == 2:
                f_even = 6
            elif h == 2:
                # vertical strip
                f_even = A1(w)
            elif w == 2:
                # horizontal strip
                f_even = A2(h)
            else:
                # corner
                f_even = A3(w, h)
            total_dbl = 2 * w * h
            f_odd = total_dbl - f_even
            cei = cnt_i_even[cw]
            coi = cnt_i_odd[cw]
            # sum contributions
            ans += cei * nj * f_even + coi * nj * f_odd

    # print final answer
    print(ans)


if __name__ == "__main__":
    main()