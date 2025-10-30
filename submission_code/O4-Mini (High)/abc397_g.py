import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))
    edges = []
    g_rev = [[] for _ in range(N)]
    g_fwd = [[] for _ in range(N)]
    for _ in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        edges.append((u, v))
        g_fwd[u].append(v)
        g_rev[v].append(u)
    # BFS from s=0 to get ds
    from collections import deque
    INF_DIST = 10**9
    ds = [INF_DIST]*N
    dq = deque([0])
    ds[0] = 0
    while dq:
        u = dq.popleft()
        du = ds[u] + 1
        for w in g_fwd[u]:
            if ds[w] > du:
                ds[w] = du
                dq.append(w)
    # BFS from t=N-1 on reverse to get dt
    dt = [INF_DIST]*N
    dq = deque([N-1])
    dt[N-1] = 0
    while dq:
        u = dq.popleft()
        du = dt[u] + 1
        for w in g_rev[u]:
            if dt[w] > du:
                dt[w] = du
                dq.append(w)
    dist0 = ds[N-1]
    # Dinic implementation
    class Dinic:
        __slots__ = ('n','g','level','it')
        def __init__(self, n):
            self.n = n
            self.g = [[] for _ in range(n)]
        def add_edge(self, u, v, c):
            # forward edge index = len(self.g[u])
            # backward edge index = len(self.g[v])
            self.g[u].append([v, c, len(self.g[v])])
            self.g[v].append([u, 0, len(self.g[u]) - 1])
        def max_flow(self, s, t):
            flow = 0
            n = self.n
            g = self.g
            # BFS to build level graph
            while True:
                level = [-1]*n
                q = deque([s])
                level[s] = 0
                while q:
                    u = q.popleft()
                    for v, cap, rev in g[u]:
                        if cap > 0 and level[v] < 0:
                            level[v] = level[u] + 1
                            if v == t:
                                break
                            q.append(v)
                if level[t] < 0:
                    break
                self.level = level
                # pointers for DFS
                it = [0]*n
                self.it = it
                # DFS send flow
                def dfs(u, f):
                    if u == t:
                        return f
                    for i in range(it[u], len(g[u])):
                        v, cap, rev = g[u][i]
                        if cap > 0 and level[v] == level[u] + 1:
                            # try push
                            ret = dfs(v, min(f, cap))
                            if ret:
                                # update capacities
                                g[u][i][1] -= ret
                                g[v][g[u][i][2]][1] += ret
                                return ret
                        it[u] += 1
                    return 0
                # repeatedly send flow
                while True:
                    pushed = dfs(s, 10**18)
                    if not pushed:
                        break
                    flow += pushed
            return flow

    answer = 0
    # Try D from dist0 down to 1
    # If no path (dist0 infinite), answer is 0
    if dist0 < INF_DIST:
        for D in range(dist0, 0, -1):
            # compute domain bounds
            # a[v] = minimal d[v], b[v] = maximal d[v]
            # a[v] = max(0, D - dt[v]), b[v] = min(ds[v], D)
            a = [0]*N
            b = [0]*N
            ok_domain = True
            for v in range(N):
                # dt[v] may be INF_DIST
                av = D - dt[v]
                if av < 0:
                    av = 0
                # if dt[v] is INF_DIST, av negative => 0
                bv = ds[v]
                if bv > D:
                    bv = D
                # ds[v] INF_DIST => bv = D
                if av > bv:
                    ok_domain = False
                    break
                a[v] = av
                b[v] = bv
            if not ok_domain:
                continue
            # build flow network
            # nodes (i,v) with i=1..D, v=0..N-1 => index = (i-1)*N + v
            # total nodes = N*D + 2
            ND = N*D + 2
            S = N*D
            T = N*D + 1
            flow = Dinic(ND)
            # infinite capacity
            INF = M*D + 5
            # domain arcs
            for v in range(N):
                av = a[v]; bv = b[v]
                # for each level i from 1..D
                # if i <= av => l[v,i]=1 => s->node infinite
                # if i > bv => l[v,i]=0 => node->t infinite
                # else free
                base = v  # we will compute idx = (i-1)*N + v
                for i in range(1, D+1):
                    idx = (i-1)*N + base
                    if i <= av:
                        # must be 1
                        flow.add_edge(S, idx, INF)
                    elif i > bv:
                        # must be 0
                        flow.add_edge(idx, T, INF)
                    # else optional
            # monotonic constraints: for v, for i=1..D-1: (v,i+1)->(v,i) infinite
            for v in range(N):
                base = v
                for i in range(1, D):
                    uidx = i*N + base      # (i+1)-1 = i
                    vidx = (i-1)*N + base # (i)-1 = i-1
                    flow.add_edge(uidx, vidx, INF)
            # cost arcs: for each original edge u->v, for each i, add (v,i)->(u,i) cap=1
            for (u, v) in edges:
                # for i in 1..D
                # idx_v = (i-1)*N + v, idx_u = (i-1)*N + u
                # arc from idx_v to idx_u with cap=1
                # unconditionally
                for i in range(D):
                    vidx = i*N + v
                    uidx = i*N + u
                    flow.add_edge(vidx, uidx, 1)
            # compute min-cut = max-flow
            mf = flow.max_flow(S, T)
            if mf <= K:
                answer = D
                break
    # print result
    print(answer)

if __name__ == "__main__":
    main()