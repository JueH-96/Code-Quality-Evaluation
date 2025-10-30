import sys
import itertools

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    INF = 10**18
    dist = [[INF] * (n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dist[i][i] = 0

    bridges = [None]
    for i in range(1, m+1):
        u = int(next(it)); v = int(next(it)); t = int(next(it))
        bridges.append((u, v, t))
        if t < dist[u][v]:
            dist[u][v] = t
            dist[v][u] = t

    for k in range(1, n+1):
        for i in range(1, n+1):
            if dist[i][k] == INF:
                continue
            for j in range(1, n+1):
                new_dist = dist[i][k] + dist[k][j]
                if new_dist < dist[i][j]:
                    dist[i][j] = new_dist

    q = int(next(it))
    out_lines = []
    for _ in range(q):
        k = int(next(it))
        bridge_ids = [int(next(it)) for _ in range(k)]
        edges = []
        for bid in bridge_ids:
            u, v, t = bridges[bid]
            edges.append((u, v, t))
        
        if k == 0:
            out_lines.append(str(dist[1][n]))
            continue
            
        best = INF
        for perm in itertools.permutations(range(k)):
            for mask in range(1 << k):
                current = 1
                total = 0
                for i in range(k):
                    idx = perm[i]
                    u, v, w = edges[idx]
                    if mask & (1 << i):
                        s, t = v, u
                    else:
                        s, t = u, v
                    total += dist[current][s]
                    total += w
                    current = t
                total += dist[current][n]
                if total < best:
                    best = total
        out_lines.append(str(best))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()