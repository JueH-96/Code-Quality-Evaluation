import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    T = int(next(it))
    out = []
    for _ in range(T):
        K = int(next(it))
        sx = int(next(it)); sy = int(next(it))
        tx = int(next(it)); ty = int(next(it))
        # compute start cell (i1,j1) and stripe index k1
        # i1 = floor((sx+0.5)/K)
        # use integer arithmetic: x1 = 2*sx+1, cell width2 = 2*K
        x1 = sx*2 + 1
        y1 = sy*2 + 1
        w2 = K * 2
        i1 = x1 // w2
        j1 = y1 // w2
        # parity of start cell
        p1 = (i1 + j1) & 1
        # compute k1
        if p1 == 0:
            # horizontal stripe: k1 = floor((sy+0.5 - j1*K))
            # = (2*sy+1 - 2*j1*K)//2
            k1 = (y1 - (j1 * w2)) // 2
        else:
            # vertical stripe
            k1 = (x1 - (i1 * w2)) // 2
        # compute target
        x2 = tx*2 + 1
        y2 = ty*2 + 1
        i2 = x2 // w2
        j2 = y2 // w2
        p2 = (i2 + j2) & 1
        if p2 == 0:
            k2 = (y2 - (j2 * w2)) // 2
        else:
            k2 = (x2 - (i2 * w2)) // 2
        # delta
        dx = i2 - i1
        dy = j2 - j1
        Hx = dx if dx>=0 else -dx
        Hy = dy if dy>=0 else -dy
        # if same cell
        if Hx == 0 and Hy == 0:
            # no cell moves, just adjust within cell
            out.append(str(abs(k1 - k2)))
            continue
        # total minimal needed moves H0 = Hx + Hy
        H0 = Hx + Hy
        # best k1->endpoint costs
        # cost_neg = cost if first dir negative (i.e. endpoint e1=0)
        # cost_pos = cost if first dir positive (e1 = K-1)
        # negative means dir1 <0, pos means dir1 >0
        cost_neg = k1
        cost_pos = (K - 1 - k1)
        # best over both
        best_c_full = cost_neg if cost_neg < cost_pos else cost_pos
        # type of first move
        # M1 = 'H' if p1==1 else 'V'
        isHfirst = (p1 == 1)
        # Precompute bound_base: minimal H' >= H0 satisfying count bounds
        # For count bounds:
        # if first type H (p1=1): need ceil(H'/2)>=Hx => H'>=2*Hx-1 if Hx>0 else >=0
        #                       need floor(H'/2)>=Hy => H'>=2*Hy
        # if first type V (p1=0): need floor(H'/2)>=Hx => H'>=2*Hx
        #                       need ceil(H'/2)>=Hy => H'>=2*Hy-1 if Hy>0 else >=0
        if isHfirst:
            # p1==1
            if Hx > 0:
                b1 = 2 * Hx - 1
            else:
                b1 = 0
            b2 = 2 * Hy
        else:
            # p1==0
            b1 = 2 * Hy - 1 if Hy > 0 else 0
            b2 = 2 * Hx
        # base bound for counts and H0
        bound_base = H0
        if b1 > bound_base:
            bound_base = b1
        if b2 > bound_base:
            bound_base = b2
        # adjust bound_base for parity H'%2 == H0%2
        parity0 = H0 & 1
        # if parity mismatch, bump by 1
        if (bound_base & 1) != parity0:
            bound_base += 1
        # Define feasibility test for H'
        def feasible(H):
            # H >= 1 assumed
            s1 = H & 1
            if isHfirst:
                # h_count = ceil(H/2), v_count = floor(H/2)
                # ceil(H/2) = (H+s1)//2 ; floor(H/2) = (H-s1)//2
                h_count = (H + s1) // 2
                v_count = (H - s1) // 2
            else:
                # h_count = floor(H/2), v_count = ceil(H/2)
                h_count = (H - s1) // 2
                v_count = (H + s1) // 2
            # check counts
            if h_count < Hx or v_count < Hy:
                return False
            # parity assignment must allow integer plus/minus moves
            # require (h_count + dx) even and (v_count + dy) even
            # dx,dy may be negative but parity check works
            if ((h_count + dx) & 1) != 0:
                return False
            if ((v_count + dy) & 1) != 0:
                return False
            return True
        # find minimal H_L = minimal H' >= bound_base with H'%2 = parity0 and feasible
        H_L = bound_base
        # iterate attempts
        # usually within few steps
        # but make safe small loop with break
        # incremental by 2 to maintain parity
        while True:
            # H_L >=1
            if feasible(H_L):
                break
            H_L += 2
        # compute allowed signs and best cost at H_L
        # recompute h_count, v_count
        s1 = H_L & 1
        if isHfirst:
            h_count_L = (H_L + s1) // 2
            v_count_L = (H_L - s1) // 2
        else:
            h_count_L = (H_L - s1) // 2
            v_count_L = (H_L + s1) // 2
        # compute x_plus, x_minus, y_plus, y_minus
        # x_plus = (#H moves + sign moves +) = (h_count + dx)/2
        # x_minus = (h_count - dx)/2
        # dx may negative, formula holds
        # Same for y
        # All divisions must result integer if feasible
        x_plus_L = (h_count_L + dx) // 2
        x_minus_L = (h_count_L - dx) // 2
        y_plus_L = (v_count_L + dy) // 2
        y_minus_L = (v_count_L - dy) // 2
        # allowed signs for first move
        # M1 type H if isHfirst else V
        best_c1_L = None
        if isHfirst:
            # first is H slot
            # if x_plus_L>=1, positive dir allowed => cost_pos
            if x_plus_L >= 1:
                best_c1_L = cost_pos
            # if x_minus_L>=1, negative allowed => cost_neg
            if x_minus_L >= 1:
                # pick min among allowed
                if best_c1_L is None or cost_neg < best_c1_L:
                    best_c1_L = cost_neg
        else:
            # first is V slot
            if y_plus_L >= 1:
                best_c1_L = cost_pos
            if y_minus_L >= 1:
                if best_c1_L is None or cost_neg < best_c1_L:
                    best_c1_L = cost_neg
        # best_c1_L must be non-null because H_L feasible => capacity for actual moves
        # total cost for H_L
        cost_L = H_L + best_c1_L
        # if best allowed at H_L gives global best c1, we can finish
        if best_c1_L == best_c_full:
            # best possible k cost used
            out.append(str(cost_L))
            continue
        # else we attempt flex variant
        # sign0 is which direction gives c_best
        # if cost_neg < cost_pos => use negative sign0 else positive
        if cost_neg <= cost_pos:
            # negative best
            want_neg = True
        else:
            want_neg = False
        # compute bound2 for flex bound
        if isHfirst:
            # M1=H, require x_minus>=1 or x_plus>=1 for opposite sign, adjusting generic
            # bound_base already >= count_base
            # need ceil(H'/2)>=Hx+2 => H'>=2*(Hx+2)-1
            bound2 = 2 * Hx + 3
        else:
            # M1=V, need ceil(H'/2)>=Hy+2 => H'>=2*(Hy+2)-1
            bound2 = 2 * Hy + 3
        # overall flex bound
        if bound2 > bound_base:
            bound_flex = bound2
        else:
            bound_flex = bound_base
        # adjust for parity
        if (bound_flex & 1) != parity0:
            bound_flex += 1
        # now find minimal H_flex >= bound_flex, step by 2
        H_f = bound_flex
        while True:
            # check feasible and capacity for want_neg/want_pos
            if feasible(H_f):
                # recompute counts
                s1 = H_f & 1
                if isHfirst:
                    h_c = (H_f + s1) // 2
                    v_c = (H_f - s1) // 2
                else:
                    h_c = (H_f - s1) // 2
                    v_c = (H_f + s1) // 2
                # compute slot sign counts
                x_p = (h_c + dx) // 2
                x_m = (h_c - dx) // 2
                y_p = (v_c + dy) // 2
                y_m = (v_c - dy) // 2
                ok = False
                if isHfirst:
                    if want_neg:
                        if x_m >= 1:
                            ok = True
                    else:
                        if x_p >= 1:
                            ok = True
                else:
                    if want_neg:
                        if y_m >= 1:
                            ok = True
                    else:
                        if y_p >= 1:
                            ok = True
                if ok:
                    break
            H_f += 2
        cost_F = H_f + best_c_full
        # result is best of two
        ans = cost_L if cost_L < cost_F else cost_F
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()