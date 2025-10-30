import sys
sys.setrecursionlimit(10**7)
readline = sys.stdin.readline
INF = 10**10

class BellmanFord:
    def __init__(self, V: int):
        self.V = V
        self.E = []
    
    def add_arc(self, fr: int, to: int, cost: int):
        self.E.append((fr, to, cost))
    
    def shortest_path(self, s: int) -> list:
        self.d = [INF] * self.V
        self.prev = [0] * self.V
        self.negative_cycle = False
        self.d[s] = 0
        for i in range(self.V):
            for f, t, c in self.E:
                if self.d[t] > self.d[f] + c:
                    self.d[t] =  self.d[f] + c
                    self.prev[t] = f
                    if i + 1 == self.V:
                        self.negative_cycle = True
        if self.negative_cycle:
            self.get_negative_cycle(s)

    def get_negative_cycle(self, s: int):
        s = self.prev[s]
        negative_cycle = [s]
        while True:
            s = self.prev[s]
            if s in negative_cycle:
                negative_cycle.append(s)
                negative_cycle = negative_cycle[negative_cycle.index(s):]
                break
            negative_cycle.append(s)
        self.negative_cycle = negative_cycle[::-1]
    
    def is_negative_cycle(self) -> bool:
        return self.negative_cycle
                
    
    def shortest_path_to_index(self):
        shortest_paths_dict = {s:self.shortest_path(s) for s in range(self.V)}
        return shortest_paths_dict
    
    def print(self, s: int, t: int):
        self.shortest_path(s)
        if self.d[t] < INF:
            print(self.d[t], end=" ")
            i = t
            while i != -1:
                print(f"<{self.prev[i]}, {i}>", end=" ")
                i = self.prev[i]
            print("")

class SCC(object):
    def __init__(self, G):
        V = len(G)
        self.G = G
        self.GT = [[] for _ in range(V)]
        self.order = []
        self.used = [False] * V
 
    def dfs(self, v):
        self.used[v] = True
        for next_ in self.G[v]:
            if not self.used[next_]:
                self.dfs(next_)
        self.order.append(v)
 
    def rdfs(self, v, color_id):
        self.color[v] = color_id
        for next_ in self.GT[v]:
            if self.color[next_] == -1:
                self.rdfs(next_, color_id)
 
    def construct(self):
        V = len(self.G)
        self.color = [-1] * V
 
        # first pass: generate finish order
        for v in range(V):
            if not self.used[v]:
                self.dfs(v)
 
        # second pass: generate scc
        color_id = 0
        for v in self.order[::-1]:
            if self.color[v] == -1:
                self.rdfs(v, color_id)
                color_id += 1
 
        return self.color

class BipartiteMatching:
    def __init__(self, N: int, M: int, AB: list) -> None:
        self.N = N
        self.M = M
        self.AB = AB
        self.G = [[] for _ in range(N)]
        for a, b in AB:
            self.G[a].append(b)
            
        self.new_pos = [b + N for a, b in self.AB]
        self.g = SCC(self.build_residual_graph()).construct()
        self.matchings = {}
        for i, m in enumerate(self.g[:N]):
            self.matchings[i] = -1
            for j in range(len(self.G[i])):
                if self.g[i] == self.g[self.G[i][j] + N]:
                    self.matchings[i] = self.G[i][j]
        
    def build_residual_graph(self):
        g = [[] for _ in range(self.N + self.M)]
        for i in range(self.N):
            g[i].append((self.new_pos[i], 0, 1))
            for j in range(len(self.G[i])):
                u = self.G[i][j]
                g[i].append((u + self.N, 1, 0))
                g[u + self.N].append((i, 0, 1))
                
        return g
    
    def is_matching(self):
        return len(self.matchings) == len(self.re})

N, M = list(map(int, input().split()))
g = [[] for _ in range(N)]
re = [[] for _ in range(N)]
edges = []
for _ in range(M):
    a, b = list(map(int, input().split()))
    a -= 1
    b -= 1
    g[a].append(b)
    re[b].append(a)
    edges.append((a, b))

bf = BellmanFord(N)
for fr, to in edges:
    bf.add_arc(fr, to, 1)
bf.shortest_path(0)
if bf.is_negative_cycle():
    print(-1)
else:
    g = [[] for _ in range(N)]
    for fr, to in edges:
        if bf.d[fr] + 1 == bf.d[to]:
            g[fr].append(to)
    K = BellmanFord(N)
    for ed in edges:
        K.add_arc(ed[1], ed[0], 1)
    K.shortest_path(0)
    g_rev = [[] for _ in range(N)]
    for fr, to in edges:
        if K.d[to] + 1 == K.d[fr]:
            g_rev[fr].append(to)
    edges = [(a, b) for a, b in edges if a in g[0] or b in g_rev[0]]

    ans = 10**10
    for a, b in edges:
        M = BipartiteMatching(N, N, edges)
        for i, ma in M.matchings.items():
            if ma == -1: continue
            if i == a and ma == b:
                M.matchings[i] = -1
                a_to_b = sum([1 for mm in M.matchings.values() if mm != -1]) * 2 + 1
                M.matchings[i] = b
                M.matchings[i] = -1
                b_to_a = sum([1 for mm in M.matchings.values() if mm != -1]) * 2 + 1
                M.matchings[i] = b
                ans = min(ans, a_to_b , b_to_a)
    print(ans if ans < 10**9 else -1)