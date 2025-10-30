import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    k = int(data[1])
    total_vertices = n * k
    edges = []
    index = 2
    graph = [[] for _ in range(total_vertices)]
    for i in range(total_vertices - 1):
        u = int(data[index]); v = int(data[index+1]); index += 2
        u -= 1
        v -= 1
        graph[u].append(v)
        graph[v].append(u)
    
    if k == 1:
        print("Yes")
        return
        
    parent = [-1] * total_vertices
    children = [[] for _ in range(total_vertices)]
    q = deque([0])
    parent[0] = -1
    while q:
        u = q.popleft()
        for v in graph[u]:
            if v == parent[u]:
                continue
            parent[v] = u
            children[u].append(v)
            q.append(v)
    
    bfs_order = []
    queue = deque([0])
    while queue:
        u = queue.popleft()
        bfs_order.append(u)
        for v in children[u]:
            queue.append(v)
    rev_order = bfs_order[::-1]
    
    ret = [0] * total_vertices
    
    for u in rev_order:
        fail = False
        chains = []
        for v in children[u]:
            if ret[v] == -1:
                fail = True
                break
            elif ret[v] > 0:
                chains.append(ret[v])
                
        if fail:
            ret[u] = -1
            continue
            
        if len(chains) > 2:
            ret[u] = -1
            continue
            
        if len(chains) == 0:
            ret[u] = 1
        elif len(chains) == 1:
            a = chains[0]
            if a + 1 == k:
                ret[u] = 0
            else:
                if a + 1 < k:
                    ret[u] = a + 1
                else:
                    ret[u] = -1
        else:
            a, b = chains
            if a + b + 1 == k:
                ret[u] = 0
            else:
                ret[u] = -1
                
    if ret[0] == 0:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()