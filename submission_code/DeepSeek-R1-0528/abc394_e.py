import collections

def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0])
    graph = []
    for i in range(1, 1 + n):
        graph.append(data[i].strip())
    
    in_edges = [[] for _ in range(n)]
    out_edges = [[] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            c = graph[i][j]
            if c != '-':
                out_edges[i].append((j, c))
                in_edges[j].append((i, c))
                
    dist = [[-1] * n for _ in range(n)]
    q = collections.deque()
    
    for i in range(n):
        dist[i][i] = 0
        q.append((i, i))
        
    for i in range(n):
        for j in range(n):
            if graph[i][j] != '-':
                if dist[i][j] == -1:
                    dist[i][j] = 1
                    q.append((i, j))
                    
    while q:
        u, v = q.popleft()
        for x, c1 in in_edges[u]:
            for y, c2 in out_edges[v]:
                if c1 == c2:
                    if dist[x][y] == -1:
                        dist[x][y] = dist[u][v] + 2
                        q.append((x, y))
                        
    for i in range(n):
        print(" ".join(map(str, dist[i])))
        
if __name__ == '__main__':
    main()