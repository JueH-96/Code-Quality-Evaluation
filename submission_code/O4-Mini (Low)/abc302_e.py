import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    # adjacency list: using sets for O(1) removals
    adj = [set() for _ in range(N+1)]
    # degrees
    deg = [0] * (N+1)
    # count of isolated vertices (degree zero)
    isolated = N

    out = []
    for _ in range(Q):
        query = list(map(int, input().split()))
        t = query[0]
        if t == 1:
            # add edge u-v
            u, v = query[1], query[2]
            # if u was isolated, it now becomes non-isolated
            if deg[u] == 0:
                isolated -= 1
            # if v was isolated, it now becomes non-isolated
            if deg[v] == 0:
                isolated -= 1
            # add edge
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        else:
            # remove all edges incident to v
            v = query[1]
            if deg[v] > 0:
                # v becomes isolated
                isolated += 1
                # for each neighbor u, remove v-u
                for u in adj[v]:
                    adj[u].remove(v)
                    deg[u] -= 1
                    # if u becomes isolated now
                    if deg[u] == 0:
                        isolated += 1
                # clear v's adjacency
                adj[v].clear()
                deg[v] = 0
        out.append(str(isolated))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()