import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    from collections import deque

    data = sys.stdin.readline
    N_M = data().split()
    if not N_M:
        return
    N = int(N_M[0])
    M = int(N_M[1])
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u_v = data().split()
        if not u_v:
            u_v = data().split()
        u = int(u_v[0]) - 1
        v = int(u_v[1]) - 1
        adj[u].append(v)
        adj[v].append(u)

    # color[v] = -1 if unvisited; else 0 or 1 = side in bipartition
    color = [-1] * N

    found_odd_comp = False
    # for even-size comps, accumulate parity of side0 count
    side0_parity = 0

    # BFS/DFS to find connected components and bipartition counts
    for i in range(N):
        if color[i] != -1:
            continue
        # new component
        # BFS using deque
        dq = deque()
        color[i] = 0
        dq.append(i)
        c0 = 1
        c1 = 0
        comp_size = 1
        while dq:
            u = dq.popleft()
            cu = color[u]
            for w in adj[u]:
                if color[w] == -1:
                    # assign opposite color
                    cw = cu ^ 1
                    color[w] = cw
                    if cw == 0:
                        c0 += 1
                    else:
                        c1 += 1
                    comp_size += 1
                    dq.append(w)
                # else: already visited; input guarantees bipartite, so no conflict check
        # record this component
        if comp_size & 1:
            found_odd_comp = True
        else:
            # even-size comp: its side0 count parity is fixed
            side0_parity ^= (c0 & 1)

    # Decide winner
    # First player = Aoki, second = Takahashi
    if N & 1:
        # N odd: total moves parity = M mod2
        if M & 1:
            print("Aoki")
        else:
            print("Takahashi")
    else:
        # N even
        if found_odd_comp:
            # if any odd-size component: Aoki can force parity
            print("Aoki")
        else:
            # all comps even-size: parity fixed by side0_parity
            # total moves parity = side0_parity - M mod2 = (side0_parity ^ (M&1))
            # first wins if that = 1
            if side0_parity ^ (M & 1):
                print("Aoki")
            else:
                print("Takahashi")

if __name__ == "__main__":
    main()