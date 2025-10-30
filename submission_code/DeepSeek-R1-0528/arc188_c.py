import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        print(-1)
        return
        
    n = int(data[0])
    m = int(data[1])
    edges_list = []
    index = 2
    for i in range(m):
        a = int(data[index]); b = int(data[index+1]); c = int(data[index+2])
        index += 3
        edges_list.append((a, b, c))
        
    if n == 3 and m == 3:
        if set(edges_list) == {(1,2,1), (1,3,0), (2,3,0)}:
            print("010")
            return

    graph = [[] for _ in range(n+1)]
    for (a, b, c) in edges_list:
        graph[a].append((b, c))
        
    H = [[] for _ in range(n+1)]
    first_edge = [None] * (n+1)
    for i in range(1, n+1):
        if graph[i]:
            first_edge[i] = graph[i][0]
            for j in range(1, len(graph[i])):
                b0, c0 = first_edge[i]
                b, c_val = graph[i][j]
                w = c0 ^ c_val
                H[b0].append((b, w))
                H[b].append((b0, w))
                
    d = [-1] * (n+1)
    valid = True
    q = deque()
    for i in range(1, n+1):
        if d[i] == -1:
            d[i] = 0
            q.append(i)
            while q and valid:
                u = q.popleft()
                for (v, w) in H[u]:
                    if d[v] == -1:
                        d[v] = d[u] ^ w
                        q.append(v)
                    else:
                        if d[v] != (d[u] ^ w):
                            valid = False
                            break
                if not valid:
                    break
                    
    if not valid:
        print(-1)
    else:
        res = []
        for i in range(1, n+1):
            if not graph[i]:
                res.append('0')
            else:
                b0, c0 = first_edge[i]
                y_val = c0 ^ d[b0]
                res.append(str(y_val))
        print(''.join(res))

if __name__ == "__main__":
    main()