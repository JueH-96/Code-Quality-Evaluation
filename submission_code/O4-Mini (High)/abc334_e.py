import sys
def main():
    import sys
    data = sys.stdin.readline().split()
    H = int(data[0]); W = int(data[1])
    comp = [None] * H
    A = 0
    # Read grid, mark green as -2 (unvisited), red as -1
    for i in range(H):
        line = sys.stdin.readline().rstrip()
        # -2 for '#', -1 for '.'
        row = [(-2 if ch == '#' else -1) for ch in line]
        comp[i] = row
        # count red cells
        A += line.count('.')
    # Flood‐fill to label green components
    C0 = 0
    C = comp
    H_ = H; W_ = W
    for i in range(H_):
        row_i = C[i]
        for j in range(W_):
            if row_i[j] == -2:
                cid = C0
                # mark start
                row_i[j] = cid
                # DFS stack
                si = [i]
                sj = [j]
                while si:
                    ti = si.pop(); tj = sj.pop()
                    row_t = C[ti]
                    # up
                    if ti > 0 and C[ti-1][tj] == -2:
                        C[ti-1][tj] = cid
                        si.append(ti-1); sj.append(tj)
                    # down
                    if ti+1 < H_ and C[ti+1][tj] == -2:
                        C[ti+1][tj] = cid
                        si.append(ti+1); sj.append(tj)
                    # left
                    if tj > 0 and row_t[tj-1] == -2:
                        row_t[tj-1] = cid
                        si.append(ti); sj.append(tj-1)
                    # right
                    if tj+1 < W_ and row_t[tj+1] == -2:
                        row_t[tj+1] = cid
                        si.append(ti); sj.append(tj+1)
                C0 += 1
    # Compute expected value
    mod = 998244353
    invA = pow(A, mod-2, mod)
    # Sum of distinct adjacent green components over all red cells
    S = 0
    C = comp
    for i in range(H_):
        row_cur = C[i]
        row_up  = C[i-1] if i > 0 else None
        row_dn  = C[i+1] if i+1 < H_ else None
        for j in range(W_):
            if row_cur[j] != -1:
                continue
            cnt = 0
            a = b = c = -1
            # up
            if row_up is not None:
                u = row_up[j]
                if u >= 0:
                    a = u; cnt = 1
            # down
            if row_dn is not None:
                u = row_dn[j]
                if u >= 0:
                    if cnt == 0:
                        a = u; cnt = 1
                    elif u != a:
                        b = u; cnt = 2
            # left
            if j > 0:
                u = row_cur[j-1]
                if u >= 0:
                    if cnt == 0:
                        a = u; cnt = 1
                    elif cnt == 1:
                        if u != a:
                            b = u; cnt = 2
                    else:
                        if u != a and u != b:
                            c = u; cnt += 1
            # right
            if j+1 < W_:
                u = row_cur[j+1]
                if u >= 0:
                    if cnt == 0:
                        a = u; cnt = 1
                    elif cnt == 1:
                        if u != a:
                            b = u; cnt = 2
                    elif cnt == 2:
                        if u != a and u != b:
                            c = u; cnt = 3
                    else:
                        if u != a and u != b and u != c:
                            cnt += 1
            S += cnt
    # expected = C0 + (A - S)/A = C0 + 1 - S/A
    ans = ((C0 + 1) % mod - (S % mod) * invA % mod) % mod
    sys.stdout.write(str(ans))

main()