import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    adj = [[] for _ in range(n+1)]
    index = 1
    for i in range(n-1):
        u = int(data[index])
        v = int(data[index+1])
        index += 2
        adj[u].append(v)
        adj[v].append(u)
    
    parent_arr = [0] * (n+1)
    size_arr = [0] * (n+1)
    children = [[] for _ in range(n+1)]
    stack = [1]
    visited = [0] * (n+1)
    visited[1] = 1
    order = []
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = 1
                parent_arr[v] = u
                children[u].append(v)
                stack.append(v)
    
    for i in range(len(order)-1, -1, -1):
        u = order[i]
        size_arr[u] = 1
        for v in children[u]:
            size_arr[u] += size_arr[v]
    
    unmatched = [[] for _ in range(n+1)]
    matches = [0] * (n+1)
    for i in range(len(order)-1, -1, -1):
        u = order[i]
        lst = [u]
        for v in children[u]:
            lst.extend(unmatched[v])
        x = min(size_arr[u], n - size_arr[u])
        to_remove = len(lst) - x
        if to_remove % 2 != 0:
            to_remove -= 1
        for j in range(0, to_remove, 2):
            a = lst[j]
            b = lst[j+1]
            matches[a] = b
            matches[b] = a
        unmatched[u] = lst[to_remove:]
    
    deg = [0] * (n+1)
    for i in range(1, n+1):
        deg[i] = len(adj[i])
    
    q = deque()
    for i in range(1, n+1):
        if deg[i] == 1:
            q.append(i)
    
    removed = [0] * (n+1)
    ans = []
    while q:
        u = q.popleft()
        if removed[u]:
            continue
        v = matches[u]
        if removed[v]:
            continue
        if deg[v] == 1:
            ans.append((u, v))
            removed[u] = 1
            removed[v] = 1
            for w in adj[u]:
                if not removed[w]:
                    deg[w] -= 1
                    if deg[w] == 1:
                        q.append(w)
            for w in adj[v]:
                if not removed[w]:
                    deg[w] -= 1
                    if deg[w] == 1:
                        q.append(w)
    
    for u, v in ans:
        print(u, v)

if __name__ == '__main__':
    main()