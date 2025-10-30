import collections
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    graph = [[] for _ in range(n+1)]
    candidate_nodes = set()
    index = 2
    for _ in range(m):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        graph[a].append(b)
        if b == 1:
            candidate_nodes.add(a)
    
    dist = [-1] * (n+1)
    q = collections.deque()
    dist[1] = 0
    q.append(1)
    
    while q:
        u = q.popleft()
        for v in graph[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
                
    ans = float('inf')
    for u in candidate_nodes:
        if dist[u] != -1:
            if dist[u] + 1 < ans:
                ans = dist[u] + 1
                
    print(-1 if ans == float('inf') else ans)

if __name__ == "__main__":
    main()