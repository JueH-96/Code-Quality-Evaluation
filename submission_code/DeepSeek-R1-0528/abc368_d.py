import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    k = int(data[1])
    graph = [[] for _ in range(n+1)]
    index = 2
    for _ in range(n-1):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        graph[a].append(b)
        graph[b].append(a)
    
    terms = list(map(int, data[index:index+k]))
    is_term = [False] * (n+1)
    for t in terms:
        is_term[t] = True

    deg = [0] * (n+1)
    for i in range(1, n+1):
        deg[i] = len(graph[i])
    
    removed_node = [False] * (n+1)
    q = deque()
    for i in range(1, n+1):
        if deg[i] == 1 and not is_term[i]:
            q.append(i)
    
    removed = 0
    while q:
        u = q.popleft()
        removed_node[u] = True
        removed += 1
        for v in graph[u]:
            if not removed_node[v]:
                deg[v] -= 1
                if deg[v] == 1 and not is_term[v]:
                    q.append(v)
    
    print(n - removed)

if __name__ == "__main__":
    main()