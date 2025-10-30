import sys
import threading
import heapq

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N = int(input().strip())
    C = [input().strip() for _ in range(N)]

    # Build outgoing and incoming adjacency lists by letter
    # out_edges[u][c] = list of v such that there's an edge u->v labeled c
    # in_edges[v][c] = list of u such that there's an edge u->v labeled c
    out_edges = [dict() for _ in range(N)]
    in_edges = [dict() for _ in range(N)]
    for u in range(N):
        for v in range(N):
            ch = C[u][v]
            if ch != '-':
                out_edges[u].setdefault(ch, []).append(v)
                in_edges[v].setdefault(ch, []).append(u)

    # We'll compute the minimum palindrome-path length between all pairs (i,j)
    # by running Dijkstra over the "state graph" whose nodes are pairs (u,v).
    # Transitions:
    #   - From (u,u) to itself with cost=0 (initial)
    #   - If there is an edge u->v with label c, then (u,v) has cost=1 from start
    #   - From (u,v), for any letter c, for any u' in out_edges[u][c] and any v' in in_edges[v][c],
    #     we can go to (u',v') with additional cost=2.

    INF = 10**18
    dist = [[INF]*N for _ in range(N)]
    pq = []

    # Initialize distances:
    # (i,i) = 0
    for i in range(N):
        dist[i][i] = 0
        heapq.heappush(pq, (0, i, i))

    # (u,v) = 1 if there's a direct edge u->v
    for u in range(N):
        for ch, vs in out_edges[u].items():
            for v in vs:
                if dist[u][v] > 1:
                    dist[u][v] = 1
                    heapq.heappush(pq, (1, u, v))

    # Dijkstra
    while pq:
        d, u, v = heapq.heappop(pq)
        if d > dist[u][v]:
            continue
        # Try to extend palindrome by matching letters on both ends
        # i.e., pick a letter c, go u->u' on c and v'<-v on c
        # cost +2
        # For each letter present in both out_edges[u] and in_edges[v]
        # iterate that letter's lists
        # We iterate over the smaller dict to reduce overhead
        if len(out_edges[u]) < len(in_edges[v]):
            # iterate over out_edges[u]
            for ch, u_succs in out_edges[u].items():
                if ch not in in_edges[v]:
                    continue
                v_preds = in_edges[v][ch]
                newd = d + 2
                for u2 in u_succs:
                    row = dist[u2]
                    for v2 in v_preds:
                        if row[v2] > newd:
                            row[v2] = newd
                            heapq.heappush(pq, (newd, u2, v2))
        else:
            # iterate over in_edges[v]
            for ch, v_preds in in_edges[v].items():
                if ch not in out_edges[u]:
                    continue
                u_succs = out_edges[u][ch]
                newd = d + 2
                for u2 in u_succs:
                    row = dist[u2]
                    for v2 in v_preds:
                        if row[v2] > newd:
                            row[v2] = newd
                            heapq.heappush(pq, (newd, u2, v2))

    # Print answers
    out = []
    for i in range(N):
        line = []
        for j in range(N):
            ans = dist[i][j]
            if ans >= INF:
                ans = -1
            line.append(str(ans))
        out.append(" ".join(line))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()