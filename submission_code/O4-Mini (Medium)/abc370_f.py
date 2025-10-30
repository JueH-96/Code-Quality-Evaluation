import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.read().split()
    N = int(data[0]); K = int(data[1])
    A = list(map(int, data[2:]))

    S = sum(A)
    # build next1 for given x
    def build_next1(x):
        # prefix sum
        ps = [0] * (2*N+1)
        for i in range(2*N):
            ps[i+1] = ps[i] + A[i % N]
        nxt = [0] * (2*N+1)
        r = 0
        for i in range(2*N):
            if r < i:
                r = i
            # move r until sum >= x
            while r < 2*N and ps[r+1] - ps[i] < x:
                r += 1
            # now either r==2N or ps[r+1]-ps[i]>=x
            # we want smallest j such that sum[i:j]>=x: that's j=r+1 if valid, else 2N
            if r < 2*N and ps[r+1] - ps[i] >= x:
                nxt[i] = r+1
            else:
                nxt[i] = 2*N
        nxt[2*N] = 2*N
        return nxt

    # build jump table
    def build_jump(nxt):
        maxlg = (K.bit_length())
        jump = [nxt]
        for b in range(1, maxlg):
            prev = jump[b-1]
            curr = [0] * len(prev)
            for i in range(len(prev)):
                j = prev[i]
                curr[i] = prev[j] if j < len(prev) else len(prev)-1
            jump.append(curr)
        return jump

    # compute p_K[s] using jump table
    def get_pK(s, jump):
        pos = s
        b = 0
        kk = K
        while kk:
            if kk & 1:
                pos = jump[b][pos]
                if pos >= 2*N:
                    return 2*N
            kk >>= 1
            b += 1
        return pos

    # check feasibility for x: exists s in [0,N) such that p_K[s] <= s+N
    def feasible(x):
        nxt = build_next1(x)
        jump = build_jump(nxt)
        # scan s
        lim = 2*N
        for s in range(0, N):
            end = get_pK(s, jump)
            if end <= s + N:
                return True
        return False

    # binary search x
    lo = 0; hi = S // K + 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    x = lo

    # recompute for x to get s_min, s_max, and jump table
    nxt = build_next1(x)
    jump = build_jump(nxt)
    s_min = None; s_max = None
    # also record pK[s] to avoid recompute?
    for s in range(0, N):
        end = get_pK(s, jump)
        if end <= s + N:
            if s_min is None:
                s_min = s
            s_max = s
    # now build p_j sequences for s_min and s_max
    # p_j are the endpoints after each jump
    p_min = [0] * K
    p_max = [0] * K
    # simulate jumps
    pos = s_min
    for j in range(K):
        nxtp = nxt[pos]
        p_min[j] = nxtp
        pos = nxtp
    pos = s_max
    for j in range(K):
        nxtp = nxt[pos]
        p_max[j] = nxtp
        pos = nxtp

    # build intervals mod N
    intervals = []
    full_cover = False
    for j in range(K):
        l = p_min[j]
        r = p_max[j]
        if r - l + 1 >= N:
            full_cover = True
            break
        lm = l % N
        rm = r % N
        if lm <= rm:
            intervals.append((lm, rm))
        else:
            # wraps
            intervals.append((lm, N-1))
            intervals.append((0, rm))
    if full_cover:
        y = 0
    else:
        # merge intervals
        intervals.sort()
        covered = 0
        cur_l = -1; cur_r = -1
        for l, r in intervals:
            if cur_l == -1:
                cur_l, cur_r = l, r
            else:
                if l > cur_r + 1:
                    covered += (cur_r - cur_l + 1)
                    cur_l, cur_r = l, r
                else:
                    if r > cur_r:
                        cur_r = r
        if cur_l != -1:
            covered += (cur_r - cur_l + 1)
        y = N - covered

    # output
    print(x, y)


if __name__ == "__main__":
    main()