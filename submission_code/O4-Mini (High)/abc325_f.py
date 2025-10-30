import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    D = [0]*N
    for i in range(N):
        D[i] = int(next(it))
    L1 = int(next(it)); C1 = int(next(it)); K1 = int(next(it))
    L2 = int(next(it)); C2 = int(next(it)); K2 = int(next(it))
    # Choose sensor X as the one with smaller available count
    if K1 <= K2:
        Kx, Lx, Cx = K1, L1, C1
        Ky, Ly, Cy = K2, L2, C2
    else:
        Kx, Lx, Cx = K2, L2, C2
        Ky, Ly, Cy = K1, L1, C1

    # Precompute, for each section, the list of (dx, dy) options:
    # using dx sensors of type X, dy sensors of type Y needed (dy <= Ky).
    f_lists = []
    INF = Ky + 1
    for Di in D:
        # maximum X sensors that could help this section
        # we only consider dx up to min(Kx, ceil(Di/Lx))
        # compute ceil(Di/Lx)
        if Di <= 0:
            max_x = 0
        else:
            q = Di // Lx
            if q * Lx < Di:
                q += 1
            max_x = q
        if max_x > Kx:
            max_x = Kx
        flist = []
        last_f = -1
        # enumerate dx from 0..max_x
        # compute f = ceil(max(0, Di - dx*Lx) / Ly)
        # skip if f > Ky, and skip repeats of f
        for dx in range(max_x + 1):
            rem = Di - dx * Lx
            if rem <= 0:
                f = 0
            else:
                # ceil(rem / Ly)
                # Ly > 0
                t = rem // Ly
                if t * Ly < rem:
                    t += 1
                f = t
            if f > Ky:
                continue
            if f != last_f:
                flist.append((dx, f))
                last_f = f
            # once f has reached 0, it stays 0, skip repeats
        if not flist:
            # impossible to cover this section within sensor limits
            sys.stdout.write("-1")
            return
        f_lists.append(flist)

    # dp_old[kx] = minimal number of Y sensors needed to cover processed sections using kx X-sensors
    Kx_local = Kx
    INF_local = INF
    # initialize dp for 0 sections
    dp_old = [INF_local] * (Kx_local + 1)
    dp_old[0] = 0

    # Process each section
    for flist in f_lists:
        dp_prev = dp_old
        # new dp
        dp_new = [INF_local] * (Kx_local + 1)
        # handle dx=0 option if present
        # find index in flist where dx == 0
        idx0 = 0
        if flist[0][0] == 0:
            # use this as base
            dx0, dy0 = flist[0]
            if dy0 == 0:
                # dp_new[k] = dp_prev[k]
                for k in range(Kx_local + 1):
                    v = dp_prev[k]
                    dp_new[k] = v if v < INF_local else INF_local
            else:
                # dp_new[k] = dp_prev[k] + dy0 (clamped)
                # only if sum <= Ky
                lim = INF_local - dy0
                for k in range(Kx_local + 1):
                    v = dp_prev[k]
                    if v <= lim:
                        dp_new[k] = v + dy0
                    else:
                        # dp_new[k] stays INF_local
                        pass
            idx0 = 1
        # for other (dx, dy)
        # localize for speed
        dpn = dp_new
        dpv = dp_prev
        INFv = INF_local
        Kxv = Kx_local
        # iterate other options
        # for each (dx,dy) with dx>0
        for j in range(idx0, len(flist)):
            dx, dy = flist[j]
            # dx > 0 guaranteed here
            # only consider dx <= Kxv
            # maximum old index is Kxv-dx
            max_old = Kxv - dx
            if max_old < 0:
                break
            # we can also prune if dy > Ky, but flist built dy <=Ky
            # iterate old kx
            # v_old < INFv implies reachable
            # v_new = v_old + dy; if v_new < dpn[k_new] and v_new < INFv: update
            lim = INFv - dy
            # local references
            dpn_loc = dpn
            dpv_loc = dpv
            for k_old in range(max_old + 1):
                v_old = dpv_loc[k_old]
                if v_old <= lim:
                    v_new = v_old + dy
                    k_new = k_old + dx
                    # compare and assign
                    if v_new < dpn_loc[k_new]:
                        dpn_loc[k_new] = v_new
        dp_old = dp_new

    # dp_old now has minimal Y sensors needed for kx = 0..Kx
    # compute minimal cost
    ans = None
    # local cost constants
    cost_x = Cx
    cost_y = Cy
    for kx in range(Kx_local + 1):
        y_used = dp_old[kx]
        if y_used <= Ky:
            cost = kx * cost_x + y_used * cost_y
            if ans is None or cost < ans:
                ans = cost
    if ans is None:
        sys.stdout.write("-1")
    else:
        sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()