import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    graph = [[] for _ in range(n+1)]
    rev_graph = [[] for _ in range(n+1)]
    
    index = 2
    for _ in range(m):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        graph[a].append(b)
        rev_graph[b].append(a)
    
    INF = 10**18
    d1 = [INF] * (n+1)
    d1[1] = 0
    q = deque([1])
    while q:
        u = q.popleft()
        for v in graph[u]:
            if d1[v] > d1[u] + 1:
                d1[v] = d1[u] + 1
                q.append(v)
    
    d2 = [INF] * (n+1)
    d2[1] = 0
    q = deque([1])
    while q:
        u = q.popleft()
        for v in rev_graph[u]:
            if d2[v] > d2[u] + 1:
                d2[v] = d2[u] + 1
                q.append(v)
    
    ans = INF
    for u in range(2, n+1):
        if d1[u] != INF and d2[u] != INF:
            total = d1[u] + d2[u]
            if total < ans:
                ans = total
    
    print(ans if ans != INF else -1)

if __name__ == "__main__":
    main()