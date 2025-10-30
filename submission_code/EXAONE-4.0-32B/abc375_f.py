import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    it = iter(data)
    n = int(next(it)); m = int(next(it)); q = int(next(it))
    edges = []
    for i in range(m):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        c = int(next(it))
        edges.append((a, b, c))
    
    queries = []
    removal_set = set()
    for i in range(q):
        t = next(it)
        if t == '1':
            idx = int(next(it)) - 1
            queries.append(('remove', idx))
            removal_set.add(idx)
        else:
            x = int(next(it)) - 1
            y = int(next(it)) - 1
            queries.append(('query', x, y))
    
    INF = 10**18
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        
    for i in range(m):
        if i in removal_set:
            continue
        a, b, c = edges[i]
        if c < dist[a][b]:
            dist[a][b] = c
            dist[b][a] = c
            
    for k in range(n):
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                new_dist = dist[i][k] + dist[k][j]
                if new_dist < dist[i][j]:
                    dist[i][j] = new_dist
                    
    ans = []
    for i in range(q-1, -1, -1):
        if queries[i][0] == 'query':
            x, y = queries[i][1], queries[i][2]
            if dist[x][y] >= INF:
                ans.append(-1)
            else:
                ans.append(dist[x][y])
        else:
            idx = queries[i][1]
            a, b, c = edges[idx]
            if c < dist[a][b]:
                dist[a][b] = c
                dist[b][a] = c
            for k in [a, b]:
                for i in range(n):
                    if dist[i][k] == INF:
                        continue
                    for j in range(n):
                        new_dist = dist[i][k] + dist[k][j]
                        if new_dist < dist[i][j]:
                            dist[i][j] = new_dist
                            
    for i in range(len(ans)-1, -1, -1):
        print(ans[i])

if __name__ == '__main__':
    main()