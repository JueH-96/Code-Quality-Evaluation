from collections import deque
import sys

def main():
    N1, N2, M = map(int, sys.stdin.readline().split())
    first_adj = [[] for _ in range(N1 + 1)]
    second_adj = [[] for _ in range(N1 + N2 + 1)]
    
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        if a <= N1 and b <= N1:
            first_adj[a].append(b)
            first_adj[b].append(a)
        else:
            second_adj[a].append(b)
            second_adj[b].append(a)
    
    def bfs(start, adj, max_node):
        dist = [-1] * (max_node + 1)
        q = deque()
        dist[start] = 0
        q.append(start)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return max(dist[1:max_node+1])
    
    d1_max = bfs(1, first_adj, N1)
    d2_max = bfs(N1 + N2, second_adj, N1 + N2)
    print(d1_max + d2_max + 1)

if __name__ == "__main__":
    main()