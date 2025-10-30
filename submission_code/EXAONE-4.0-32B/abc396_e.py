import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    
    graph = [[] for _ in range(n+1)]
    
    for _ in range(m):
        x = int(next(it))
        y = int(next(it))
        z = int(next(it))
        if x == y:
            if z != 0:
                print(-1)
                return
        else:
            graph[x].append((y, z))
            graph[y].append((x, z))
    
    visited = [False] * (n+1)
    base = [0] * (n+1)
    result = [0] * (n+1)
    
    for i in range(1, n+1):
        if not visited[i]:
            q = deque([i])
            visited[i] = True
            comp_nodes = [i]
            valid_component = True
            
            while q:
                u = q.popleft()
                for v, w in graph[u]:
                    if not visited[v]:
                        visited[v] = True
                        base[v] = base[u] ^ w
                        comp_nodes.append(v)
                        q.append(v)
                    else:
                        if base[u] ^ base[v] != w:
                            print(-1)
                            return
            
            n_nodes = len(comp_nodes)
            bit_counts = [0] * 31
            for node in comp_nodes:
                num = base[node]
                for j in range(31):
                    if num & (1 << j):
                        bit_counts[j] += 1
            
            a_comp = 0
            for j in range(31):
                ones = bit_counts[j]
                zeros = n_nodes - ones
                if ones > zeros:
                    a_comp |= (1 << j)
            
            for node in comp_nodes:
                result[node] = a_comp ^ base[node]
    
    print(' '.join(str(result[i]) for i in range(1, n+1)))

if __name__ == "__main__":
    main()