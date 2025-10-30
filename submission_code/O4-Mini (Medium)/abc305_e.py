import sys
import threading
import heapq

def main():
    import sys
    data = sys.stdin
    # Read N, M, K
    line = data.readline().split()
    N = int(line[0])
    M = int(line[1])
    K = int(line[2])
    # Build adjacency list
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        a, b = map(int, data.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    # best[v] = maximum remaining stamina upon reaching v (or -1 if unvisited)
    best = [-1] * (N+1)
    # Priority queue for multi-source BFS by descending stamina
    # We store (-remain, vertex) so that largest remain is popped first.
    pq = []
    for _ in range(K):
        p, h = map(int, data.readline().split())
        # If multiple guards at same vertex, we'd keep the larger stamina; here p's are distinct
        best[p] = h
        heapq.heappush(pq, (-h, p))
    # BFS-like propagation
    while pq:
        neg_rem, v = heapq.heappop(pq)
        rem = -neg_rem
        # If this entry is stale, skip
        if best[v] != rem:
            continue
        if rem == 0:
            continue  # no further spread
        # Spread to neighbors
        nxt_rem = rem - 1
        for u in adj[v]:
            if nxt_rem > best[u]:
                best[u] = nxt_rem
                heapq.heappush(pq, (-nxt_rem, u))
    # Gather all guarded vertices: best[v] >= 0
    guarded = [str(v) for v in range(1, N+1) if best[v] >= 0]
    # Output
    out = sys.stdout
    out.write(str(len(guarded)) + "
")
    if guarded:
        out.write(" ".join(guarded) + "
")

if __name__ == "__main__":
    main()