import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    mod = 998244353
    data = sys.stdin.read().split()
    H = int(data[0]); W = int(data[1])
    S = data[2:]
    # Transpose to make width <= height
    if W > H:
        # transpose grid
        T = [''.join(S[i][j] for i in range(H)) for j in range(W)]
        S = T
        H, W = W, H
    # Now H >= W, and W*H <= 200, width W <= 14
    # Precompute all horizontal-proper patterns of length W
    pats = []       # digits list
    codes = []      # packed 2-bit codes
    # mapping digit to 2-bit: '1'->0, '2'->1, '3'->2
    # but here we just generate all
    max_code = 1 << (2*W)
    # We'll generate by DFS
    def dfs_build(pos, last, digs, code):
        if pos == W:
            pats.append(digs[:])
            codes.append(code)
            return
        for d in (0,1,2):
            if d == last: continue
            digs.append(d)
            dfs_build(pos+1, d, digs, code | (d << (2*pos)))
            digs.pop()
    dfs_build(0, -1, [], 0)
    N = len(pats)
    # Precompute for each row the list of pattern indices matching fixed cells
    row_valid = []
    for i in range(H):
        row = S[i]
        vr = []
        # for each pattern idx, check match
        for idx in range(N):
            digs = pats[idx]
            ok = True
            for j,ch in enumerate(row):
                if ch != '?' and (digs[j] != (ord(ch)-ord('1'))):
                    ok = False; break
            if ok:
                vr.append(idx)
        # if any row has zero valid patterns => answer 0
        if not vr:
            print(0)
            return
        row_valid.append(vr)
    # Precompute bit masks for compatibility test
    # low bits at even positions, high bits at odd positions
    mask_low = 0
    mask_high = 0
    for k in range(W):
        mask_low |= (1 << (2*k))
        mask_high |= (1 << (2*k+1))
    # DP arrays
    dp_prev = [0] * N
    # row 0 init
    for idx in row_valid[0]:
        dp_prev[idx] = 1
    prev_list = row_valid[0]
    # iterate rows 1..H-1
    for i in range(1, H):
        vr = row_valid[i]
        dp_cur = [0] * N
        # localize for speed
        dp_p = dp_prev
        codelist = codes
        ml = mask_low; mh = mask_high
        for cur in vr:
            code_c = codelist[cur]
            s = 0
            # sum over prev_list
            for pr in prev_list:
                # test compatibility
                x = codelist[pr] ^ code_c
                neg = ~x
                # conflict if any 2-bit segment zero => check bits
                if ( (neg & ml) & ((neg & mh) >> 1) ) == 0:
                    s += dp_p[pr]
            dp_cur[cur] = s % mod
        # prepare next
        dp_prev = dp_cur
        prev_list = vr
    # sum up last row
    ans = 0
    for idx in prev_list:
        ans = (ans + dp_prev[idx]) % mod
    print(ans)

if __name__ == "__main__":
    main()