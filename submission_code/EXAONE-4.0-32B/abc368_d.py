import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    
    graph = [[] for _ in range(n+1)]
    for _ in range(n-1):
        u = int(next(it))
        v = int(next(it))
        graph[u].append(v)
        graph[v].append(u)
        
    marked_vertices = []
    for _ in range(k):
        marked_vertices.append(int(next(it)))
    
    special = [False] * (n+1)
    for node in marked_vertices:
        special[node] = True
        
    total_marked = k
    
    parent = [0] * (n+1)
    children = [[] for _ in range(n+1)]
    stack = [1]
    parent[1] = 0
    order = []
    
    while stack:
        u = stack.pop()
        order.append(u)
        for v in graph[u]:
            if v == parent[u]:
                continue
            parent[v] = u
            children[u].append(v)
            stack.append(v)
            
    count = [0] * (n+1)
    for i in range(len(order)-1, -1, -1):
        u = order[i]
        if special[u]:
            count[u] = 1
        else:
            count[u] = 0
        for child in children[u]:
            count[u] += count[child]
            
    total_edges = 0
    for u in range(2, n+1):
        if count[u] > 0 and (total_marked - count[u]) > 0:
            total_edges += 1
            
    print(total_edges + 1)

if __name__ == "__main__":
    main()