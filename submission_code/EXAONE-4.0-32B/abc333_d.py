import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    graph = [[] for _ in range(n+1)]
    idx = 1
    for _ in range(n-1):
        u = int(data[idx])
        v = int(data[idx+1])
        idx += 2
        graph[u].append(v)
        graph[v].append(u)
    
    parent = [0] * (n+1)
    size = [1] * (n+1)
    
    q = deque([1])
    parent[1] = 0
    order = [1]
    while q:
        u = q.popleft()
        for v in graph[u]:
            if v == parent[u]:
                continue
            parent[v] = u
            q.append(v)
            order.append(v)
            
    for i in range(len(order)-1, -1, -1):
        u = order[i]
        if parent[u] != 0:
            size[parent[u]] += size[u]
            
    max_child_size = 0
    for child in graph[1]:
        if size[child] > max_child_size:
            max_child_size = size[child]
            
    ans = n - max_child_size
    print(ans)

if __name__ == '__main__':
    main()