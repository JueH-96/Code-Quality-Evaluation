import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    graph = [[] for _ in range(n+1)]
    index = 2
    for _ in range(m):
        x = int(data[index])
        y = int(data[index+1])
        z = int(data[index+2])
        index += 3
        graph[x].append((y, z))
        graph[y].append((x, z))
    
    rel = [None] * (n+1)
    ans = [0] * (n+1)
    
    for i in range(1, n+1):
        if rel[i] is None:
            q = deque([i])
            rel[i] = 0
            comp_nodes = [i]
            while q:
                u = q.popleft()
                for (v, w) in graph[u]:
                    if rel[v] is None:
                        rel[v] = rel[u] ^ w
                        q.append(v)
                        comp_nodes.append(v)
                    else:
                        if rel[u] ^ rel[v] != w:
                            print(-1)
                            return
            comp_vals = [rel[node] for node in comp_nodes]
            k = 0
            for j in range(31):
                count0 = 0
                count1 = 0
                for val in comp_vals:
                    if (val >> j) & 1:
                        count1 += 1
                    else:
                        count0 += 1
                if count1 > count0:
                    k |= (1 << j)
            for node in comp_nodes:
                ans[node] = k ^ rel[node]
                
    print(" ".join(str(ans[i]) for i in range(1, n+1)))

if __name__ == "__main__":
    main()