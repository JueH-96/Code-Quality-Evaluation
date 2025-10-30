import sys
from collections import deque

def main():
    input = sys.stdin.readline
    N = int(input())
    A = [0] + list(map(int, input().split()))
    # state: 0=unvisited,1=visiting,2=visited
    state = [0] * (N + 1)
    idx = [-1] * (N + 1)
    in_cycle = [False] * (N + 1)
    # csize[u] will store the size of the cycle that u eventually reaches
    csize = [0] * (N + 1)

    for i in range(1, N + 1):
        if state[i] != 0:
            continue
        u = i
        path = []
        # Walk until we hit a visited node
        while True:
            if state[u] == 0:
                # mark visiting
                state[u] = 1
                idx[u] = len(path)
                path.append(u)
                u = A[u]
            elif state[u] == 1:
                # found a cycle
                start = idx[u]
                cycle_nodes = path[start:]
                sz = len(cycle_nodes)
                for v in cycle_nodes:
                    in_cycle[v] = True
                    csize[v] = sz
                break
            else:
                # state[u] == 2, we've reached an already processed region
                break
        # mark all in path as fully visited
        for v in path:
            state[v] = 2
            idx[v] = -1

    # build reverse graph
    rev = [[] for _ in range(N + 1)]
    for u in range(1, N + 1):
        rev[A[u]].append(u)

    # BFS from cycle nodes to assign dist and propagate cycle size
    dist = [-1] * (N + 1)
    q = deque()
    for u in range(1, N + 1):
        if in_cycle[u]:
            dist[u] = 0
            q.append(u)
    while q:
        u = q.popleft()
        for v in rev[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                csize[v] = csize[u]
                q.append(v)

    # sum up reachable counts
    ans = 0
    for u in range(1, N + 1):
        # reachable_count[u] = dist[u] + csize[u]
        ans += dist[u] + csize[u]
    print(ans)

main()