import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, Q = map(int, input().split())
    # degree of each vertex
    deg = [0] * (N + 1)
    # adjacency sets
    adj = [set() for _ in range(N + 1)]
    # number of isolated vertices (degree zero)
    isolated = N

    out = []
    for _ in range(Q):
        q = input().split()
        t = q[0]
        if t == '1':
            u = int(q[1]); v = int(q[2])
            # add edge u-v
            if deg[u] == 0:
                isolated -= 1
            if deg[v] == 0:
                isolated -= 1
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        else:  # t == '2'
            v = int(q[1])
            if deg[v] > 0:
                # v will become isolated
                isolated += 1
            # remove all edges incident to v
            for u in adj[v]:
                # remove the reverse edge
                adj[u].remove(v)
                deg[u] -= 1
                if deg[u] == 0:
                    isolated += 1
            # clear v's adjacency
            adj[v].clear()
            deg[v] = 0

        # record answer
        out.append(str(isolated))

    # output all at once
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()