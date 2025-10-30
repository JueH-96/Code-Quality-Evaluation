import sys
import threading
from collections import deque

def main():
    import sys
    input = sys.stdin.readline
    N = int(input().strip())
    # fwd[u]: dict label->list of v such that u->v has that label
    # rev[v]: dict label->list of u such that u->v has that label
    fwd = [dict() for _ in range(N)]
    rev = [dict() for _ in range(N)]
    for u in range(N):
        line = input().rstrip('
')
        for v, ch in enumerate(line):
            if ch != '-':
                # forward edge u->v labeled ch
                fwd[u].setdefault(ch, []).append(v)
                # reverse adjacency for v
                rev[v].setdefault(ch, []).append(u)
    # dist[u][v] = shortest palindrome-path length from u to v, or -1
    dist = [[-1]*N for _ in range(N)]
    dq = deque()
    # Base: empty path for u->u
    for i in range(N):
        dist[i][i] = 0
        dq.append((i, i))
    # Base: single-edge palindromes
    for u in range(N):
        for ch, vs in fwd[u].items():
            for v in vs:
                if dist[u][v] == -1:
                    dist[u][v] = 1
                    dq.append((u, v))
    # BFS over state-space of pairs (u,v)
    while dq:
        u, v = dq.popleft()
        d = dist[u][v]
        # We try to extend palindrome at both ends by matching labels:
        # pick an incoming edge i->u labeled c, and an outgoing edge v->j labeled c
        # then we get a palindrome from i to j of length d+2
        # rev[u]: incoming to u, fwd[v]: outgoing from v
        incoming = rev[u]
        outgoing = fwd[v]
        # iterate over labels present in incoming
        for c, from_list in incoming.items():
            # if same label c leaves v
            if c in outgoing:
                to_list = outgoing[c]
                nd = d + 2
                for i_node in from_list:
                    # we want dist[i_node][j_node]
                    row_i = dist[i_node]
                    for j_node in to_list:
                        if row_i[j_node] == -1:
                            row_i[j_node] = nd
                            dq.append((i_node, j_node))
    # Output
    out = []
    for i in range(N):
        row = dist[i]
        # convert to strings
        out.append(" ".join(str(x) for x in row))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()