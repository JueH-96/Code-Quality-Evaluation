import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input_data = sys.stdin.buffer.readline

    N, Q = map(int, input_data().split())

    # adjacency lists (sets) for every vertex 1 … N
    adj = [set() for _ in range(N + 1)]

    # degree of every vertex
    deg = [0] * (N + 1)

    # number of isolated vertices (degree == 0)
    isolated = N

    out_lines = []

    for _ in range(Q):
        parts = input_data().split()
        if parts[0] == b'1':                       # add edge
            u = int(parts[1])
            v = int(parts[2])

            if deg[u] == 0:
                isolated -= 1
            if deg[v] == 0:
                isolated -= 1

            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        else:                                     # remove all edges incident to v
            v = int(parts[1])

            if deg[v]:                            # only if there is something to remove
                neighbours = list(adj[v])         # copy because we modify sets
                for u in neighbours:
                    adj[u].remove(v)
                    deg[u] -= 1
                    if deg[u] == 0:
                        isolated += 1

                adj[v].clear()
                deg[v] = 0
                isolated += 1                      # v itself becomes isolated

        out_lines.append(str(isolated))

    sys.stdout.write('
'.join(out_lines))


if __name__ == "__main__":
    main()