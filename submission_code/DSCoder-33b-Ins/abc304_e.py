from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def is_reachable(self, s, d):
        visited = [False] * (self.V)
        queue = []
        queue.append(s)
        visited[s] = True
        while queue:
            n = queue.pop(0)
            if n == d:
                return True
            for i in self.graph[n]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
        return False

def solve():
    N, M = map(int, input().split())
    g = Graph(N)
    for _ in range(M):
        u, v = map(int, input().split())
        g.add_edge(u-1, v-1)
    K = int(input())
    for _ in range(K):
        x, y = map(int, input().split())
        g.add_edge(x-1, y-1)
    Q = int(input())
    for _ in range(Q):
        p, q = map(int, input().split())
        if g.is_reachable(p-1, q-1):
            print("No")
        else:
            print("Yes")
            g.add_edge(p-1, q-1)

solve()