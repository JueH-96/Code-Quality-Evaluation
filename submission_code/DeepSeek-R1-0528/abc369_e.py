import itertools
import sys

INF = 10**18

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    all_bridges = []
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        t = int(next(it))
        all_bridges.append((u, v, t))
    
    dist = [[INF] * (n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dist[i][i] = 0
        
    for u, v, t_val in all_bridges:
        if t_val < dist[u][v]:
            dist[u][v] = t_val
            dist[v][u] = t_val
            
    for k in range(1, n+1):
        dk = dist[k]
        for i in range(1, n+1):
            if dist[i][k] == INF:
                continue
            di = dist[i]
            dik = di[k]
            for j in range(1, n+1):
                new_dist = dik + dk[j]
                if new_dist < di[j]:
                    di[j] = new_dist
                    
    q = int(next(it))
    out_lines = []
    for _ in range(q):
        k_i = int(next(it))
        bridge_ids = [int(next(it)) for _ in range(k_i)]
        req_bridges = [all_bridges[idx-1] for idx in bridge_ids] if k_i > 0 else []
        
        if not req_bridges:
            ans = dist[1][n]
            out_lines.append(str(ans))
            continue
            
        ans = INF
        for perm in itertools.permutations(req_bridges):
            num_bridges = len(perm)
            for mask in range(1 << num_bridges):
                total = 0
                u0, v0, t0 = perm[0]
                if mask & 1:
                    a0 = v0
                    b0 = u0
                else:
                    a0 = u0
                    b0 = v0
                total += dist[1][a0] + t0
                
                prev_b = b0
                for j in range(1, num_bridges):
                    uj, vj, tj = perm[j]
                    bit = (mask >> j) & 1
                    if bit:
                        aj = vj
                        bj = uj
                    else:
                        aj = uj
                        bj = vj
                    total += dist[prev_b][aj] + tj
                    prev_b = bj
                    
                total += dist[prev_b][n]
                if total < ans:
                    ans = total
                    
        out_lines.append(str(ans))
        
    print("
".join(out_lines))

if __name__ == '__main__':
    main()