import collections
import sys

class Dinic:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]
        self.level = [0] * n
        self.ptr = [0] * n

    def add_edge(self, u, v, cap):
        forward = [v, cap, len(self.graph[v])]
        backward = [u, 0, len(self.graph[u])]
        self.graph[u].append(forward)
        self.graph[v].append(backward)

    def bfs(self, s, t):
        self.level = [-1] * self.n
        self.level[s] = 0
        q = collections.deque([s])
        while q:
            u = q.popleft()
            for e in self.graph[u]:
                if e[1] > 0 and self.level[e[0]] < 0:
                    self.level[e[0]] = self.level[u] + 1
                    q.append(e[0])
        return self.level[t] >= 0

    def dfs(self, u, t, f):
        if u == t:
            return f
        for i in range(self.ptr[u], len(self.graph[u])):
            e = self.graph[u][i]
            if e[1] > 0 and self.level[u] + 1 == self.level[e[0]]:
                pushed = self.dfs(e[0], t, min(f, e[1]))
                if pushed > 0:
                    e[1] -= pushed
                    rev_e = self.graph[e[0]][e[2]]
                    rev_e[1] += pushed
                    return pushed
            self.ptr[u] += 1
        return 0

    def max_flow(self, s, t):
        flow = 0
        while self.bfs(s, t):
            self.ptr = [0] * self.n
            while True:
                pushed = self.dfs(s, t, 10**9)
                if pushed == 0:
                    break
                flow += pushed
        return flow

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    total_nodes = 26 * 2 + 2
    s_node = 52
    t_node = 53
    dinic = Dinic(total_nodes)
    ofs = 26
    
    for i in range(1, n + 1):
        s = data[i].strip()
        a = ord(s[0]) - ord('A')
        b = ord(s[1]) - ord('A')
        if a == b:
            dinic.add_edge(ofs + a, a, 1)
        else:
            dinic.add_edge(ofs + b, a, 1)
        dinic.add_edge(s_node, ofs + a, 1)
        dinic.add_edge(b, t_node, 1)
    
    for i in range(26):
        dinic.add_edge(i, ofs + i, 10**9)
    
    flow = dinic.max_flow(s_node, t_node)
    ans = n - flow
    print(ans)

if __name__ == "__main__":
    main()