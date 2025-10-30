import heapq
import math
from typing import List

INF = 1 << 62

class MinCostFlow:
    def __init__(self, N: int):
        self.N = N
        self.G = [[] for _ in range(N)]
        self.E = []
        
    def add_edge(self, fr: int, to: int, cap: int, cost: int):
        self.G[fr].append(len(self.E))
        self.E.append([fr, to, cap, cost])
        self.G[to].append(len(self.E))
        self.E.append([to, fr, 0, -cost])
    
    def flow(self, s: int, t: int) -> List[int]:
        def dijkstra():
            H = [INF] * self.N
            H[s] = 0
            que = [(0, s)]
            dist = [INF] * self.N
            dist[s] = 0
            while que:
                cost, v = heapq.heappop(que)
                if H[v] < cost: continue
                for id in self.G[v]:
                    nv, cap, cost = self.E[id][1], self.E[id][2], self.E[id][3]
                    if cap > 0 and H[nv] > H[v] + cost:
                        H[nv] = H[v] + cost
                        heapq.heappush(que, (H[nv], nv))
                        prev[nv] = id
                        dist[nv] = dist[v] + 1
            return dist, prev
        
        prev = [-1] * self.N
        prevedge = [-1] * self.N
        dist, prev = dijkstra()
        if dist[t] == INF:
            return None
        
        flow = 0
        cost = 0
        while dist[t] < self.N:
            f = INF
            v = t
            while v != s:
                f = min(f, self.E[prev[v]][2])
                v = self.E[prev[v]][1]
            
            v = t
            while v != s:
                id = prev[v]
                self.E[id][2] -= f
                self.E[id ^ 1][2] += f
                cost += f * self.E[id][3]
                v = self.E[id][1]
            
            flow += f
            dist, prev = dijkstra()
            if dist[t] == INF:
                break
        
        return flow, cost

    def min_cost_flow(self, s: int, t: int, f: int):
        res = 0
        while f > 0:
            mf, mc = self.flow(s, t)
            if mf is None:
                return INF
            a = min(mf, f)
            res += a * mc
            f -= a
        return res

N, X = map(int, input().split())
Ev = [list(map(int, input().split())) for i in range(N)]
A, P, B, Q = [], [], [], []
for a, p, b, q in Ev:
    if p <= q:
        A.append(a)
        P.append(p)
        B.append(0)
        Q.append(INF)
    else:
        A.append(0)
        P.append(INF)
        B.append(b)
        Q.append(q)

graph = MinCostFlow(N + 2)
s = N
t = N + 1
for i, (a, p) in enumerate(zip(A, P)):
    graph.add_edge(s, i, min(X // p, INF), 0)
    graph.add_edge(i, t, INF, p * math.ceil(INF / a))
for i, (b, q) in enumerate(zip(B, Q)):
    graph.add_edge(s, i, min(X // q, INF), 0)
    graph.add_edge(i, t, INF, q * math.ceil(INF / b))
ans = 1 << 30
for i in range(100):
    connected = [set() for _ in range(N)]
    for j in range(1 << N):
        c = []
        cost = 0
        for k in range(N):
            if j & (1 << k):
                c.append(k)
                cost += graph.E[graph.G[k][0]][2] * graph.E[graph.G[k][1]][3]
        if cost > X:
            continue
        if len(c) < N or c[-1] not in connected[c[0]]:
            res = graph.min_cost_flow(s, t, INF)
            if res < INF:
                for i in range(len(c)):
                    connected[c[i]].add(c[(i + 1) % len(c)])
                    connected[c[(i + 1) % len(c)]].add(c[i])
                product = res // cost
                ans = min(ans, product)
print(ans)