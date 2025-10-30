import sys
from collections import deque

read = sys.stdin.read
readline = sys.stdin.readline
readlines = sys.stdin.readlines

class Vertex:
    def __init__(self, n: int):
        self.offset = n * 2
        self.source = self.offset # 0:的实际位置
        self.sink = self.offset + 1 # 1:的实际位置
        self.vertices = [2] * n
        self.graph = [[] for _ in range(n * 2)]
        self.depth = [0] * (n * 2)
        self.iter = [0] * (n * 2)
        self.inf = 10**20

    def set_e(self, v1: int, v2: int) -> None:
        v1 *= 2
        v2 *= 2
        self.graph[v1].append(v2 + 1)
        self.graph[v1 + 1].append(v2)
        self.vertices[v1 % 2] = min(self.vertices[v1 % 2], v1)
        self.vertices[v1 % 2] = min(self.vertices[v1 % 2], v1 + 1)
        self.vertices[v2 % 2] = min(self.vertices[v2 % 2], v2)
        self.vertices[v2 % 2] = min(self.vertices[v2 % 2], v2 + 1)

    def bfs(self):
        n = len(self.depth)
        queue = deque([self.source])
        self.depth[self.source] = 1
        while queue:
            v_from = queue.popleft()
            for v_to in self.graph[v_from]:
                if self.depth[v_to] == 0 and v_to != self.sink:
                    self.depth[v_to] = self.depth[v_from] + 1
                    queue.append(v_to)
        if self.depth[self.sink] == 0:
            self.depth = None
            
    def dfs(self, v: int, f: int) -> int:
        if v == self.sink:
            return f
        for i in range(self.iter[v], len(self.graph[v])):
            self.iter[v] = i
            v_to = self.graph[v][i]
            if self.depth[v_to] > self.depth[v]:
                d = self.dfs(v_to, min(f, self.cap[v][i]))
                if d > 0:
                    self.cap[v][i] -= d
                    self.cap[v_to][self.rev[i]] += d
                    return d
        return 0

    def max_flow(self):
        INF = self.inf
        self.bfs()
        if self.depth is None:
            return 0
        total_flow = 0
        n = len(self.graph)
        for v, g in enumerate(self.graph):
            self.cap = []
            for v_to in g:
                c = self._cap.get((v, v_to), 0)
                r = self._cap.get((v_to, v), 0)
                self.cap.append([c, len(self.graph[v_to])])
                self.rev = len(self.cap) - 1
                self.graph[v_to].append(v)
            self.cap[v].append(INF)
        while True:
            self.iter = [0] * n
            f = self.dfs(self.source, INF)
            if f == 0:
                break
            total_flow += f
        return total_flow

    def min_cut(self):
        queue = deque([self.source])
        visited = [0] * len(self.graph)
        visited[self.source] = 1
        while queue:
            v = queue.popleft()
            if v == self.sink:
                return None
            for v_to in self.graph[v]:
                if self.cc and (not self.cc[v][self.rev]) and (not visited[v_to]):
                    visited[v_to] = 1
                    queue.append(v_to)
        min_set = set([i for i in range(0, self.vertices[0], 2) if visited[i]]) | set([i for i in range(1, self.vertices[1], 2) if visited[i]])
        return min_set
    
def main():
    N, M, *NM = map(int, read().split())
    AB = [[] for _ in range(N)]
    for i, (a, b, c) in enumerate(zip(*[iter(NM)]*3)):
        AB[a - 1].append([b - 1, i, c])
        AB[b - 1].append([a - 1, i, c ^ 1])
    
    g = Vertex(N)
    for i in range(N):
        for j, v, c in AB[i]:
            g.set_e(i, j)
    
    flow = g.max_flow() 
    if flow * 2 < N:
        cut = g.min_cut()
        ans = ['0'] * N
        for i in range(N):
            if i not in cut:
                ans[i] = '1'
        print(''.join(ans))
    else:
        print(-1)


if __name__ == '__main__':
    main()