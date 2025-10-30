def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    # Read grid
    grid = [list(next(it).strip()) for _ in range(H)]
    # Choose DP width C = min(H,W), height R = max(H,W)
    if H < W:
        # transpose
        C = H
        R = W
        pre = [[-1]*C for _ in range(R)]
        for i in range(H):
            for j in range(W):
                ch = grid[i][j]
                if ch != '?':
                    pre[j][i] = ord(ch) - 49  # '1'->0, '2'->1, '3'->2
                # else remains -1
    else:
        C = W
        R = H
        pre = [[-1]*C for _ in range(R)]
        for i in range(H):
            for j in range(W):
                ch = grid[i][j]
                if ch != '?':
                    pre[i][j] = ord(ch) - 49
    # Generate all base row patterns (no horizontal repeats) of length C
    sys.setrecursionlimit(10000)
    code3_list = []      # code in base-3, 0 <= code < 3^C
    s_vals_list = []     # corresponding bytes of length C with values 0,1,2
    tmp = [0] * C
    def dfs_build(j, prev, code3):
        if j == C:
            code3_list.append(code3)
            # store a bytes object for values
            s_vals_list.append(bytes(tmp))
            return
        # try colors 0,1,2
        for t in (0,1,2):
            if t != prev:
                tmp[j] = t
                dfs_build(j+1, t, code3*3 + t)
    dfs_build(0, -1, 0)
    ns = len(code3_list)
    # Build mapping from code3 -> index (store index+1 in array)
    from array import array
    size3 = 3 ** C
    code3_map = array('I', [0]) * size3
    for idx, c3 in enumerate(code3_list):
        code3_map[c3] = idx + 1
    # Precompute transitions in CSR form
    offsets = [0] * (ns + 1)
    trans_flat = array('I', [])
    for i in range(ns):
        offsets[i] = len(trans_flat)
        svals = s_vals_list[i]
        # DP to generate all t masks (base-3 codes) for next row
        cur_masks = [0]
        cur_prev  = [-1]
        for j in range(C):
            sj = svals[j]
            nm = []
            npv = []
            # expand partials
            # avoid "=" in loops for speed
            for k in range(len(cur_masks)):
                m = cur_masks[k]
                pv = cur_prev[k]
                # try three colors
                # must differ from pv (horizontal) and from sj (vertical)
                # inline loop
                if 0 != pv and 0 != sj:
                    nm.append(m*3 + 0); npv.append(0)
                if 1 != pv and 1 != sj:
                    nm.append(m*3 + 1); npv.append(1)
                if 2 != pv and 2 != sj:
                    nm.append(m*3 + 2); npv.append(2)
            cur_masks = nm
            cur_prev  = npv
        # record neighbors
        # map each mask to its state-index
        c3m = code3_map
        for m in cur_masks:
            # c3m[m] > 0 since it's valid
            trans_flat.append(c3m[m] - 1)
    offsets[ns] = len(trans_flat)
    # free mapping to save memory
    del code3_map
    # Precompute valid state indices for each row precolor
    valid_indices = []
    for r in range(R):
        prer = pre[r]
        vr = []
        # for each state, check match
        # use locals for speed
        C_loc = C
        for idx in range(ns):
            sv = s_vals_list[idx]
            ok = True
            # unroll small loop
            for j in range(C_loc):
                pj = prer[j]
                if pj != -1 and pj != sv[j]:
                    ok = False
                    break
            if ok:
                vr.append(idx)
        valid_indices.append(vr)
        # if no valid pattern for this row, answer is 0
        if not vr:
            print(0)
            return
    # DP over rows
    mod = 998244353
    dp_prev = [0] * ns
    # initial row r=0
    for i in valid_indices[0]:
        dp_prev[i] = 1
    # iterate rows 1..R-1
    offs = offsets  # local ref
    tf   = trans_flat
    for r in range(1, R):
        vr = valid_indices[r]
        # build allowed mask
        allow = [False] * ns
        for j in vr:
            allow[j] = True
        dp_cur = [0] * ns
        ap = allow  # local
        # for each previous state
        for i in range(ns):
            v = dp_prev[i]
            if v:
                st = offs[i]; ed = offs[i+1]
                # iterate neighbors
                for k in tf[st:ed]:
                    if ap[k]:
                        tmp = dp_cur[k] + v
                        if tmp >= mod:
                            tmp -= mod
                        dp_cur[k] = tmp
        dp_prev = dp_cur
    # result = sum(dp_prev) % mod
    res = 0
    for x in dp_prev:
        res += x
    print(res % mod)

# call main
main()