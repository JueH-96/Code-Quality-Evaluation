import sys
import threading
def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    N = int(line[0]); W = int(line[1])
    # columns 0..W-1
    cols = [[] for _ in range(W)]
    # store y and block index (1-based)
    xs = [0]*(N+1)
    ys = [0]*(N+1)
    for i in range(1, N+1):
        l = data.readline().split()
        x = int(l[0]); y = int(l[1])
        xs[i] = x-1
        ys[i] = y
        cols[x-1].append((y, i))
    # compute m_x and K_max
    K_max = float('inf')
    for x in range(W):
        m = len(cols[x])
        if m < K_max:
            K_max = m
    if K_max == 0:
        # no full lines ever, no block destroyed
        INF = 10**30
        t_destroy = [INF] * (N+1)
    else:
        # sort each column by y ascending, assign positions
        # pos_i: 0-based rank in its column
        pos = [0]*(N+1)
        for x in range(W):
            cols[x].sort(key=lambda p: p[0])
            # assign pos
            lst = cols[x]
            for k in range(len(lst)):
                _, idx = lst[k]
                pos[idx] = k
        # compute M_k for k=0..K_max-1
        M = [0]*K_max
        for x in range(W):
            lst = cols[x]
            # every column has at least K_max entries
            for k in range(K_max):
                y, _ = lst[k]
                v = y - 1
                if v > M[k]:
                    M[k] = v
        # compute T_k = M[k] + (k+1)
        T = [M[k] + (k+1) for k in range(K_max)]
        INF = 10**30
        # t_destroy for each block
        t_destroy = [INF]*(N+1)
        for i in range(1, N+1):
            p = pos[i]
            if p < K_max:
                t_destroy[i] = T[p]
            # else INF
    # answer queries
    out = []
    Q = int(data.readline().strip())
    for _ in range(Q):
        l = data.readline().split()
        t = int(l[0]); a = int(l[1])
        # exists at time t+0.5 iff t < t_destroy[a]
        if t < t_destroy[a]:
            out.append("Yes")
        else:
            out.append("No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()