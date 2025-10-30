import sys
import threading
def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    n = int(line[0]); m = int(line[1]); X = int(line[2])
    out = [[] for _ in range(n)]
    inn = [[] for _ in range(n)]
    for _ in range(m):
        u,v = data.readline().split()
        u = int(u)-1; v = int(v)-1
        out[u].append(v)
        inn[v].append(u)
    # Dijkstra on 2-layer graph: layer 0 = original, layer 1 = reversed
    N = n
    INF = 10**30
    dist = [INF] * (2*N)
    dist[0] = 0
    import heapq
    hq = [(0, 0)]
    out_ls = out; inn_ls = inn
    while hq:
        d, idx = heapq.heappop(hq)
        if d != dist[idx]:
            continue
        if idx < N:
            v = idx
            nd = d + 1
            for u in out_ls[v]:
                if dist[u] > nd:
                    dist[u] = nd
                    heapq.heappush(hq, (nd, u))
            # flip to reversed layer
            nid = idx + N
            nd2 = d + X
            if dist[nid] > nd2:
                dist[nid] = nd2
                heapq.heappush(hq, (nd2, nid))
        else:
            v = idx - N
            nd = d + 1
            for u in inn_ls[v]:
                nid = u + N
                if dist[nid] > nd:
                    dist[nid] = nd
                    heapq.heappush(hq, (nd, nid))
            # flip back to original layer
            nid = v
            nd2 = d + X
            if dist[nid] > nd2:
                dist[nid] = nd2
                heapq.heappush(hq, (nd2, nid))
    res = dist[N-1]
    t1 = dist[N-1 + N]
    if t1 < res:
        res = t1
    # print result
    sys.stdout.write(str(res))

if __name__ == "__main__":
    main()