def main():
    import sys
    from collections import deque

    input = sys.stdin.readline

    N, M, K = map(int, input().split())
    edges = [tuple(map(int, input().split())) for _ in range(M)]
    G = [[] for _ in range(N + 1)]
    for i, (u, v) in enumerate(edges):
        G[u].append((v, i))

    # We want to place K edges of weight=1 (others=0), maximizing the
    # 0–1‐BFS distance from 1 to N.  Equivalently, find the maximum D
    # such that there is an assignment of K ones so that every path
    # 1→N picks up at least D ones.  We can binary‐search D, and for
    # each candidate D ask: “is there a way to choose K edges so that
    # the 0–1‐distance ≥ D?”
    #
    # Checking feasibility for a given D is equivalent to asking whether
    # there is an assignment w_e∈{0,1}, sum w_e = K, such that every path
    # P from 1 to N has sum_{e∈P} w_e ≥ D.  By duality this can be tested
    # by sending a flow of size D in a network where capacities are the
    # variables w_e, but since w_e must be integer 0/1, we only need to
    # check if the minimum s–t cut size ≥ D when we allow up to K ones.
    #
    # Concretely: If we give every edge capacity = 1 in a unit‐capacity
    # network, the minimum cut C is the max number of edge‐disjoint
    # paths.  To ensure every s–t path uses ≥ D of our selected edges,
    # we must place at least one “charged” edge on each of D edge‐disjoint
    # paths ⇒ we must have K ≥ D*C.  Hence the maximum achievable D is
    # floor(K / C), where C = mincut(1,N) in the original graph with
    # unit capacities.
    #
    # We compute C via Dinic, then answer = min(N-1, K // C).

    # Dinic flow for unit capacities
    class Dinic:
        class Edge:
            __slots__ = ('v','cap','rev')
            def __init__(self, v, cap, rev):
                self.v = v; self.cap = cap; self.rev = rev

        def __init__(self, n):
            self.n = n
            self.g = [[] for _ in range(n)]
            self.level = [0]*n
            self.it = [0]*n

        def add_edge(self, u, v, c):
            self.g[u].append(Dinic.Edge(v, c, len(self.g[v])))
            self.g[v].append(Dinic.Edge(u, 0, len(self.g[u]) - 1))

        def bfs(self, s):
            from collections import deque
            self.level = [-1]*self.n
            q = deque([s])
            self.level[s] = 0
            while q:
                u = q.popleft()
                for e in self.g[u]:
                    if e.cap and self.level[e.v] < 0:
                        self.level[e.v] = self.level[u] + 1
                        q.append(e.v)
            return

        def dfs(self, u, t, f):
            if u == t:
                return f
            for i in range(self.it[u], len(self.g[u])):
                e = self.g[u][i]
                if e.cap and self.level[e.v] == self.level[u] + 1:
                    ret = self.dfs(e.v, t, min(f, e.cap))
                    if ret:
                        e.cap -= ret
                        self.g[e.v][e.rev].cap += ret
                        return ret
                self.it[u] += 1
            return 0

        def max_flow(self, s, t):
            flow = 0
            INF = 10**9
            while True:
                self.bfs(s)
                if self.level[t] < 0:
                    return flow
                self.it = [0]*self.n
                while True:
                    f = self.dfs(s, t, INF)
                    if not f:
                        break
                    flow += f

    # build unit‐capacity network
    mf = Dinic(N+1)
    for u,v in edges:
        mf.add_edge(u, v, 1)

    C = mf.max_flow(1, N)
    if C == 0:
        # no path => but problem guarantees path, so won't happen
        print(0)
    else:
        # answer is the largest D with K >= D*C → D = K//C, but cannot exceed N-1
        print(min((N-1), K//C))


if __name__ == "__main__":
    main()