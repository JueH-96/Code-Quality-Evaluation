import sys
from collections import deque


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N)]
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)

    degree = [len(nei) for nei in adj]

    UNKNOWN, CENTER, LEAF = 0, 1, 2
    label = [UNKNOWN] * N          # 0: unknown, 1: center, 2: leaf
    q = deque()

    def push_leaf_neighbors(v: int) -> None:
        """Whenever vertex v obtains a label, every already–labelled
           degree-2 leaf that is adjacent to v has to be re–examined,
           so we enqueue it again."""
        for nb in adj[v]:
            if label[nb] == LEAF and degree[nb] == 2:
                q.append(nb)

    def assign(v: int, kind: int) -> None:
        """Set label[v] to kind (CENTER / LEAF) if still unknown,
           then enqueue v and its degree-2 leaf neighbours."""
        if label[v] == UNKNOWN:
            label[v] = kind
            q.append(v)
            push_leaf_neighbors(v)
        # consistency is guaranteed by the statement, so no else-branch

    # 1. obvious classifications
    for v in range(N):
        if degree[v] == 1:                # free leaves
            assign(v, LEAF)
            assign(adj[v][0], CENTER)     # their neighbours are centres
        elif degree[v] >= 3:              # leaves never have deg ≥ 3
            assign(v, CENTER)

    # 2. propagate forced choices
    while q:
        v = q.popleft()
        if label[v] == CENTER:
            # every neighbour of a centre is a leaf
            for nb in adj[v]:
                assign(nb, LEAF)

        else:  # v is a leaf
            if degree[v] == 2:
                a, b = adj[v]
                cnt_c = (label[a] == CENTER) + (label[b] == CENTER)
                cnt_l = (label[a] == LEAF) + (label[b] == LEAF)
                if cnt_c == 1:            # already has its centre
                    # the other neighbour must be a leaf
                    if label[a] == UNKNOWN:
                        assign(a, LEAF)
                    if label[b] == UNKNOWN:
                        assign(b, LEAF)
                elif cnt_l == 1:          # already has a leaf neighbour
                    # the other neighbour must be the centre
                    if label[a] == UNKNOWN:
                        assign(a, CENTER)
                    if label[b] == UNKNOWN:
                        assign(b, CENTER)
                # if nothing is forced yet we wait for more information
        # whenever we just processed v, any already labelled degree-2
        # leaves next to it might have become decidable
        push_leaf_neighbors(v)

    # 3. every vertex should now be classified,
    #    but for absolute safety treat leftovers as centres
    for v in range(N):
        if label[v] == UNKNOWN:
            label[v] = CENTER

    # 4. levels of the original stars are exactly the degrees of centres
    levels = sorted(degree[v] for v in range(N) if label[v] == CENTER)
    print(' '.join(map(str, levels)))


if __name__ == "__main__":
    main()