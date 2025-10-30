import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input())
    adj = [[] for _ in range(N+1)]
    deg = [0]*(N+1)
    for _ in range(N-1):
        a,b = map(int, input().split())
        adj[a].append(b)
        adj[b].append(a)
        deg[a] += 1
        deg[b] += 1

    alive = [True]*(N+1)
    remaining = N

    from collections import deque

    def bfs_farthest(start):
        # Returns (farthest_node, parent, dist array)
        dist = [-1]*(N+1)
        par  = [-1]*(N+1)
        q = deque([start])
        dist[start] = 0
        last = start
        while q:
            u = q.popleft()
            last = u
            for w in adj[u]:
                if alive[w] and dist[w] < 0:
                    dist[w] = dist[u] + 1
                    par[w] = u
                    q.append(w)
        return last, par, dist

    out = []
    # We'll keep a "seed" for BFS that is any alive vertex.
    # We update it each round to one of the two we just removed (if still alive) or else we scan until we find an alive vertex.
    seed = 1
    for _ in range(N//2):
        # find a live seed
        while seed <= N and not alive[seed]:
            seed += 1
        # 1) from seed find one end of diameter
        A,_,_ = bfs_farthest(seed)
        # 2) from A find B and also parent pointers
        B, par, _ = bfs_farthest(A)

        # A-B is a diameter.  We remove A,B.
        out.append((A,B))
        # mark dead
        alive[A] = False
        alive[B] = False
        remaining -= 2
        # reduce degrees (not really used further for BFS but kept for correctness)
        for w in adj[A]:
            deg[w] -= 1
        for w in adj[B]:
            deg[w] -= 1
        # next seed could be A or B if one is still alive
        seed = A if alive[A] else (B if alive[B] else seed)

    # print
    w = sys.stdout.write
    for x,y in out:
        w(f"{x} {y}
")

if __name__ == "__main__":
    main()