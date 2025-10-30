import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline().strip())
    nxt = [x - 1 for x in map(int, sys.stdin.readline().split())]

    reach = [0] * N           # reach[v] = number of vertices reachable from v

    for s in range(N):
        if reach[s]:
            continue          # already processed

        path = []             # vertices visited in this walk
        pos = dict()          # vertex -> index in path
        v = s

        # walk until we meet a vertex whose reachable size is known
        # or until we re-visit a vertex inside this walk (a cycle)
        while reach[v] == 0 and v not in pos:
            pos[v] = len(path)
            path.append(v)
            v = nxt[v]

        if reach[v]:          # the walk reached an already processed vertex
            r = reach[v]
            for u in reversed(path):
                r += 1
                reach[u] = r
        else:                 # a new cycle was found
            cycle_start = pos[v]
            cycle = path[cycle_start:]
            L = len(cycle)    # length of the cycle

            # vertices on the cycle
            for u in cycle:
                reach[u] = L

            # vertices leading into the cycle
            r = L
            for u in reversed(path[:cycle_start]):
                r += 1
                reach[u] = r

    print(sum(reach))

if __name__ == "__main__":
    main()