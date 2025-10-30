import sys
from collections import deque

def main() -> None:
    sys.setrecursionlimit(1_000_000)
    rd = sys.stdin.buffer.readline

    line = rd()
    if not line:
        return
    N, M = map(int, line.split())

    adj = [[] for _ in range(N)]          # adjacency list
    for _ in range(M):
        x, y, z = map(int, rd().split())
        x -= 1
        y -= 1
        if x == y:                        # self-loop → must have z == 0
            if z != 0:
                print(-1)
                return
            continue
        adj[x].append((y, z))
        adj[y].append((x, z))

    pre = [None] * N                      # xor from (temporary) root to each node
    ans = [0] * N                         # final answer
    BIT = 31                              # 0 … 30 are enough (2^30 > 1e9)

    for s in range(N):
        if pre[s] is not None:            # already processed component
            continue

        # ---------- BFS to build component and detect contradictions ----------
        comp_nodes = []
        dq = deque([s])
        pre[s] = 0                        # root of this component
        while dq:
            u = dq.pop()
            comp_nodes.append(u)
            for v, w in adj[u]:
                expected = pre[u] ^ w
                if pre[v] is None:
                    pre[v] = expected
                    dq.append(v)
                elif pre[v] != expected:  # contradiction found
                    print(-1)
                    return

        # ---------- choose root value r that minimises sum of A_i ----------
        cnt1 = [0] * BIT                  # how many nodes have bit k = 1 in pre
        for u in comp_nodes:
            val = pre[u]
            b = 0
            while val:
                if val & 1:
                    if b < BIT:
                        cnt1[b] += 1
                val >>= 1
                b += 1

        n_comp = len(comp_nodes)
        r = 0
        for k in range(BIT):
            if cnt1[k] > n_comp - cnt1[k]:
                r |= 1 << k               # setting bit k reduces total sum

        # ---------- produce final values ----------
        for u in comp_nodes:
            ans[u] = pre[u] ^ r

    print(' '.join(map(str, ans)))


if __name__ == "__main__":
    main()