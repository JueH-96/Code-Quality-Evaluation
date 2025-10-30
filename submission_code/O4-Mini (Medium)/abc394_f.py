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

    # Mark nodes eligible as internal (degree >=4 in original)
    isInt = [False]*(N+1)
    for u in range(1, N+1):
        if deg[u] >= 4:
            isInt[u] = True

    # Build forest adjacency among internal nodes
    fadj = [[] for _ in range(N+1)]
    for u in range(1, N+1):
        if not isInt[u]:
            continue
        for v in adj[u]:
            if isInt[v]:
                fadj[u].append(v)

    visited = [False]*(N+1)
    maxI = 0  # max number of internal nodes in a valid subtree

    # Process each component in the forest of internal nodes
    for u in range(1, N+1):
        if not isInt[u] or visited[u]:
            continue
        # Build a rooted tree of this component via DFS, collect post-order
        stack = [(u, -1, 0)]  # (node, parent, state) state 0 = pre, 1 = post
        order = []
        parent = {u: -1}
        children = {}
        visited[u] = True
        while stack:
            node, par, st = stack.pop()
            if st == 0:
                # first time at node
                stack.append((node, par, 1))
                # record parent
                parent[node] = par
                # gather children
                ch = []
                for w in fadj[node]:
                    if w == par:
                        continue
                    if not visited[w]:
                        visited[w] = True
                        ch.append(w)
                        stack.append((w, node, 0))
                children[node] = ch
            else:
                # post-order
                order.append(node)

        # DP arrays
        dp0 = {}  # dp0[node]: best size if node is root of selected subtree (no parent edge)
        dp1 = {}  # dp1[node]: best size if node is included and parent is also included

        # process in post-order
        for node in order:
            # gather gains from children: if we connect child, we get dp1[child]
            gains = []
            for w in children[node]:
                # dp1[w] must have been computed
                gains.append(dp1[w])
            # sort descending
            if gains:
                gains.sort(reverse=True)
            # dp0: can take up to 4 children
            s0 = 1
            # sum top 4
            upto0 = 4
            for i in range(min(upto0, len(gains))):
                if gains[i] > 0:
                    s0 += gains[i]
                else:
                    break
            dp0[node] = s0
            # dp1: parent is connected, so only up to 3 children
            s1 = 1
            upto1 = 3
            for i in range(min(upto1, len(gains))):
                if gains[i] > 0:
                    s1 += gains[i]
                else:
                    break
            dp1[node] = s1

            # track best root-based dp0
            if dp0[node] > maxI:
                maxI = dp0[node]

    # If no internal node possible
    if maxI <= 0:
        print(-1)
        return

    # Ensure enough leaves exist: total vertices = 3*I + 2 <= N
    # So maxI <= (N-2)//3
    cap = (N - 2) // 3
    if maxI > cap:
        maxI = cap
    if maxI <= 0:
        print(-1)
        return

    # Answer is 3*I + 2
    print(3*maxI + 2)

if __name__ == "__main__":
    main()