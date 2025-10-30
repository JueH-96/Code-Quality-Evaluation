import sys
from collections import deque

class Edge:
    def __init__(self, to, rev, capacity):
        self.to = to
        self.rev = rev
        self.capacity = capacity

class Dinic:
    def __init__(self, n):
        self.size = n
        self.graph = [[] for _ in range(n)]
    
    def add_edge(self, fr, to, cap):
        forward = Edge(to, len(self.graph[to]), cap)
        backward = Edge(fr, len(self.graph[fr]), 0)
        self.graph[fr].append(forward)
        self.graph[to].append(backward)
    
    def bfs_level(self, s, t, level):
        q = deque()
        level[:] = [-1]*self.size
        level[s] = 0
        q.append(s)
        while q:
            v = q.popleft()
            for edge in self.graph[v]:
                if edge.capacity > 0 and level[edge.to] < 0:
                    level[edge.to] = level[v]+1
                    q.append(edge.to)
                    if edge.to == t:
                        return
        return
    
    def dfs_flow(self, v, t, upTo, iter_, level):
        if v == t:
            return upTo
        for i in range(iter_[v], len(self.graph[v])):
            edge = self.graph[v][i]
            if edge.capacity >0 and level[v] < level[edge.to]:
                d = self.dfs_flow(edge.to, t, min(upTo, edge.capacity), iter_, level)
                if d>0:
                    edge.capacity -=d
                    self.graph[edge.to][edge.rev].capacity +=d
                    return d
            iter_[v] +=1
        return 0
    
    def max_flow(self, s, t):
        flow = 0
        level = [-1]*self.size
        while True:
            self.bfs_level(s, t, level)
            if level[t] <0:
                return flow
            iter_ = [0]*self.size
            while True:
                f = self.dfs_flow(s, t, float('inf'), iter_, level)
                if f ==0:
                    break
                flow +=f
            level = [-1]*self.size
        return flow

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1
    edges = []
    for _ in range(M):
        u = int(input[idx]); idx +=1
        v = int(input[idx]); idx +=1
        edges.append( (u, v) )
    
    def is_possible(D):
        if D == 0:
            return True
        total_layers = D + 1
        total_nodes = N * total_layers
        dinic = Dinic(total_nodes)
        for u_original, v_original in edges:
            for layer in range(total_layers):
                fr = (u_original-1)*total_layers + layer
                to = (v_original-1)*total_layers + layer
                dinic.add_edge(fr, to, 10**18)
                if layer < D:
                    to_next = (v_original-1)*total_layers + (layer+1)
                    dinic.add_edge(fr, to_next, 1)
        source = (1-1)*total_layers + 0
        sink = (N-1)*total_layers + 0
        flow = dinic.max_flow(source, sink)
        return flow <= K
    
    low = 0
    high = K
    answer = 0
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            answer = mid
            low = mid +1
        else:
            high = mid -1
    print(answer)

if __name__ == "__main__":
    main()