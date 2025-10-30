import sys
import threading
from collections import deque

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, M = map(int, input().split())
    # adjacency: for each node, list of (neighbor, dx, dy)
    adj = [[] for _ in range(N+1)]
    for _ in range(M):
        A, B, X, Y = map(int, input().split())
        # B = A + (X, Y)
        adj[A].append((B, X, Y))
        # A = B + (-X, -Y)
        adj[B].append((A, -X, -Y))

    # to store which component each node is in
    comp = [ -1 ] * (N+1)
    # coords relative to the component's origin
    coord = [ (0,0) ] * (N+1)
    # whether a component has the reference person 1
    comp_has_ref = []

    current_comp = 0

    # First, if node 1 exists, BFS from it to give it absolute reference
    # We treat its component as having ref.
    if comp[1] == -1:
        dq = deque()
        dq.append(1)
        comp[1] = current_comp
        coord[1] = (0, 0)
        comp_has_ref.append(True)
        while dq:
            u = dq.popleft()
            ux, uy = coord[u]
            for v, dx, dy in adj[u]:
                if comp[v] == -1:
                    # coord[v] = coord[u] + (dx, dy)
                    coord[v] = (ux + dx, uy + dy)
                    comp[v] = current_comp
                    dq.append(v)
        current_comp += 1

    # Now handle all other components
    for i in range(1, N+1):
        if comp[i] != -1:
            continue
        # start a new component with arbitrary origin
        dq = deque()
        dq.append(i)
        comp[i] = current_comp
        coord[i] = (0, 0)
        comp_has_ref.append(False)
        while dq:
            u = dq.popleft()
            ux, uy = coord[u]
            for v, dx, dy in adj[u]:
                if comp[v] == -1:
                    coord[v] = (ux + dx, uy + dy)
                    comp[v] = current_comp
                    dq.append(v)
        current_comp += 1

    # Output results
    out = []
    for i in range(1, N+1):
        c = comp[i]
        if comp_has_ref[c]:
            x, y = coord[i]
            out.append(f"{x} {y}")
        else:
            out.append("undecidable")

    print("
".join(out))


if __name__ == "__main__":
    main()