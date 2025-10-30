#!/usr/bin/env python3
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
    for _ in range(M):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        edges.append((u, v))
    # We'll do a binary search on D = 0..N-1 for the largest D s.t. we can force shortest-path >= D
    # For a given D, build a layered graph with node-capacitated gadgets and compute min-cut via max-flow.
    class Dinic:
        __slots__ = ('n','g','level','it')
        def __init__(self,n):
            self.n = n
            self.g = [[] for _ in range(n)]
        def add_edge(self,u,v,c):
            # forward edge idx len(g[u]), backward idx len(g[v])
            self.g[u].append([v,c,len(self.g[v])])
            self.g[v].append([u,0,len(self.g[u])-1])
        def bfs(self,s):
            from collections import deque
            level = [-1]*self.n
            dq = deque([s])
            level[s]=0
            while dq:
                u = dq.popleft()
                for v,c,_ in self.g[u]:
                    if c>0 and level[v]<0:
                        level[v] = level[u]+1
                        dq.append(v)
            self.level = level
            return level
        def dfs(self,u,t,f):
            if u==t:
                return f
            for i in range(self.it[u], len(self.g[u])):
                self.it[u] = i
                v,c,rev = self.g[u][i]
                if c>0 and self.level[u]<self.level[v]:
                    ret = self.dfs(v,t,min(f,c))
                    if ret>0:
                        # reduce
                        self.g[u][i][1] -= ret
                        self.g[v][rev][1] += ret
                        return ret
            return 0
        def max_flow(self,s,t,flow_limit):
            flow = 0
            INF_FLOW = flow_limit+1
            while flow < flow_limit:
                level = self.bfs(s)
                if level[t] < 0:
                    break
                self.it = [0]*self.n
                while flow < flow_limit:
                    pushed = self.dfs(s,t,flow_limit-flow)
                    if pushed<=0:
                        break
                    flow += pushed
            return flow

    def feasible(D):
        # Build flow network
        # node ids:
        #  vid( v, l ) = l*N + v,  0 <= l <= D
        #  ge_in[j]  = base + j*2
        #  ge_out[j] = base + j*2 + 1
        layers = D+1
        base = N*layers
        total_nodes = base + 2*M
        mf = Dinic(total_nodes)
        INF = K + 1  # big enough so that INF > any possible min-cut
        # For each original edge j
        for j,(u,v) in enumerate(edges):
            ge_in = base + j*2
            ge_out = ge_in + 1
            # node capacity = 1 => edge ge_in->ge_out cap=1
            mf.add_edge(ge_in, ge_out, 1)
            # penalty transitions: for l=0..D-1: (u,l) -> ge_in (INF) and ge_out -> (v,l+1) (INF)
            for l in range(D):
                mf.add_edge(l*N + u, ge_in, INF)
                mf.add_edge(ge_out, (l+1)*N + v, INF)
            # free transitions: for l=0..D: (u,l) -> (v,l) cap=INF
            for l in range(layers):
                mf.add_edge(l*N + u, l*N + v, INF)
        src = 0*N + 0  # (1,0) but using 0-based indexing: v=0,l=0 => 0
        sink = D*N + (N-1)  # (N,D): v=N-1, layer=D
        # compute min-cut = max-flow
        flow = mf.max_flow(src, sink, K+1)
        # If flow <= K, can cut with <=K edges => feasible
        return flow <= K

    # binary search max D in [0..N-1]
    lo, hi = 0, N-1
    ans = 0
    while lo <= hi:
        mid = (lo + hi)//2
        if feasible(mid):
            ans = mid
            lo = mid+1
        else:
            hi = mid-1
    # ans is maximum D such that shortest path >=D achievable
    print(ans)

if __name__ == "__main__":
    main()