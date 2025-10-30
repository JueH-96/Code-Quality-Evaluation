import sys
from collections import deque
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a = int(data[ptr])
        b = int(data[ptr+1])
        adj[a].append(b)
        adj[b].append(a)
        ptr += 2
    
    log = 18
    color = [0] * (N + 1)
    depth = [0] * (N + 1)
    parent = [[-1] * (N + 1) for _ in range(log)]
    visited = [False] * (N + 1)
    q = deque()
    q.append(1)
    visited[1] = True
    parent[0][1] = -1
    depth[1] = 0
    color[1] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                parent[0][v] = u
                depth[v] = depth[u] + 1
                color[v] = color[u] ^ 1
                q.append(v)
    
    for k in range(1, log):
        for u in range(1, N+1):
            if parent[k-1][u] == -1:
                parent[k][u] = -1
            else:
                parent[k][u] = parent[k-1][parent[k-1][u]]
    
    def lca(u, v):
        if depth[u] < depth[v]:
            u, v = v, u
        for k in range(log-1, -1, -1):
            if depth[u] - (1 << k) >= depth[v]:
                u = parent[k][u]
        if u == v:
            return u
        for k in range(log-1, -1, -1):
            if parent[k][u] != parent[k][v]:
                u = parent[k][u]
                v = parent[k][v]
        return parent[0][u]
    
    def distance(u, v):
        ancestor = lca(u, v)
        return depth[u] + depth[v] - 2 * depth[ancestor]
    
    degree = [0] * (N + 1)
    for u in range(1, N+1):
        degree[u] = len(adj[u])
    
    heap0 = []
    heap1 = []
    removed = [False] * (N + 1)
    for u in range(1, N+1):
        if degree[u] == 1 and not removed[u]:
            if color[u] == 0:
                heapq.heappush(heap0, (-depth[u], u))
            else:
                heapq.heappush(heap1, (-depth[u], u))
    
    result = []
    for _ in range(N//2):
        u = -1
        while heap0:
            d_u, u = heapq.heappop(heap0)
            d_u = -d_u
            if degree[u] == 1 and not removed[u]:
                break
            u = -1
        
        v = -1
        while heap1:
            d_v, v_candidate = heapq.heappop(heap1)
            d_v = -d_v
            if degree[v_candidate] == 1 and not removed[v_candidate]:
                v = v_candidate
                break
            v = -1
        
        if u == -1 or v == -1:
            break
        
        result.append((u, v))
        removed[u] = True
        removed[v] = True
        
        for neighbor in adj[u]:
            if not removed[neighbor]:
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    if color[neighbor] == 0:
                        heapq.heappush(heap0, (-depth[neighbor], neighbor))
                    else:
                        heapq.heappush(heap1, (-depth[neighbor], neighbor))
        
        for neighbor in adj[v]:
            if not removed[neighbor]:
                degree[neighbor] -= 1
                if degree[neighbor] == 1:
                    if color[neighbor] == 0:
                        heapq.heappush(heap0, (-depth[neighbor], neighbor))
                    else:
                        heapq.heappush(heap1, (-depth[neighbor], neighbor))
    
    for pair in result:
        print(pair[0], pair[1])

if __name__ == "__main__":
    main()