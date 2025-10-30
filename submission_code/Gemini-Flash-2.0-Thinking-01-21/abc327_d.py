import collections

def solve():
    n, m = map(int, input().split())
    a_seq = list(map(int, input().split()))
    b_seq = list(map(int, input().split()))
    
    edges = collections.defaultdict(list)
    for i in range(m):
        u, v = a_seq[i], b_seq[i]
        edges[u].append(v)
        edges[v].append(u)
        
    colors = [-1] * (n + 1)
    is_bipartite = True
    
    for start_node in range(1, n + 1):
        if colors[start_node] == -1:
            queue = collections.deque([start_node])
            colors[start_node] = 0
            while queue:
                u = queue.popleft()
                for v in edges[u]:
                    if colors[v] == -1:
                        colors[v] = 1 - colors[u]
                        queue.append(v)
                    elif colors[v] == colors[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
            if not is_bipartite:
                break
                
    if is_bipartite:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()