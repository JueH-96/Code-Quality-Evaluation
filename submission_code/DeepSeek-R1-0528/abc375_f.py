import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    Q = int(next(it))
    
    INF = 10**18
    roads = []
    base_mat = [[INF] * N for _ in range(N)]
    for i in range(N):
        base_mat[i][i] = 0
        
    for i in range(M):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        c = int(next(it))
        roads.append((a, b))
        if c < base_mat[a][b]:
            base_mat[a][b] = c
            base_mat[b][a] = c

    dist = [row[:] for row in base_mat]
    for k in range(N):
        row_k = dist[k]
        for i in range(N):
            if dist[i][k] == INF:
                continue
            for j in range(N):
                new_val = dist[i][k] + row_k[j]
                if new_val < dist[i][j]:
                    dist[i][j] = new_val
                    
    out_lines = []
    for q in range(Q):
        t = next(it)
        if t == '1':
            i_road = int(next(it)) - 1
            a0, b0 = roads[i_road]
            if dist[a0][b0] == INF:
                pass
            else:
                dist[a0][b0] = INF
                dist[b0][a0] = INF
                for k in range(N):
                    row_k = dist[k]
                    for i in range(N):
                        dist_i_k = dist[i][k]
                        if dist_i_k == INF:
                            continue
                        row_i = dist[i]
                        for j in range(N):
                            s = dist_i_k + row_k[j]
                            if s < row_i[j]:
                                row_i[j] = s
        else:
            x = int(next(it)) - 1
            y = int(next(it)) - 1
            res = dist[x][y]
            out_lines.append(str(res) if res != INF else "-1")
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()