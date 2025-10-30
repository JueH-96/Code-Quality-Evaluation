import sys
from collections import deque

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    graph = [[] for _ in range(n+1)]
    for i in range(1, n):
        u, v = map(int, data[i].split())
        graph[u].append(v)
        graph[v].append(u)
    
    if n == 1:
        print(0)
        return
        
    parent = [0] * (n+1)
    children = [[] for _ in range(n+1)]
    size = [1] * (n+1)  # Each node initially has size 1 (itself)
    
    q = deque([1])
    parent[1] = 0
    order = [1]
    while q:
        u = q.popleft()
        for v in graph[u]:
            if v == parent[u]:
                continue
            parent[v] = u
            children[u].append(v)
            q.append(v)
            order.append(v)
            
    for i in range(len(order)-1, -1, -1):
        u = order[i]
        for child in children[u]:
            size[u] += size[child]
            
    max_child_size = 0
    for child in children[1]:
        if size[child] > max_child_size:
            max_child_size = size[child]
            
    print(n - max_child_size)

if __name__ == "__main__":
    main()