import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N = int(input())
    a = [0] + list(map(int, input().split()))

    # Build indegree and reverse graph
    indeg = [0] * (N + 1)
    rev = [[] for _ in range(N + 1)]
    for u in range(1, N + 1):
        v = a[u]
        indeg[v] += 1
        rev[v].append(u)

    # Kahn's algorithm to remove tree nodes, leaving cycles
    from collections import deque
    q = deque()
    for i in range(1, N + 1):
        if indeg[i] == 0:
            q.append(i)

    while q:
        u = q.popleft()
        v = a[u]
        indeg[v] -= 1
        if indeg[v] == 0:
            q.append(v)

    # Nodes with indeg>0 are in cycles
    in_cycle = [False] * (N + 1)
    for i in range(1, N + 1):
        if indeg[i] > 0:
            in_cycle[i] = True

    # Discover each cycle and assign its size
    cycle_sz = [0] * (N + 1)
    visited = [False] * (N + 1)
    for i in range(1, N + 1):
        if in_cycle[i] and not visited[i]:
            # traverse the cycle starting at i
            cnt = 0
            cur = i
            while True:
                visited[cur] = True
                cnt += 1
                cur = a[cur]
                if cur == i:
                    break
            # assign size to all in this cycle
            cur = i
            while True:
                cycle_sz[cur] = cnt
                cur = a[cur]
                if cur == i:
                    break

    # BFS from cycle nodes to assign distance to cycle and inherit cycle size
    dist = [-1] * (N + 1)
    dq = deque()
    for i in range(1, N + 1):
        if in_cycle[i]:
            dist[i] = 0
            dq.append(i)

    while dq:
        v = dq.popleft()
        for u in rev[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                cycle_sz[u] = cycle_sz[v]
                dq.append(u)

    # Sum up for all u: reachable count = dist[u] + cycle_sz[u]
    ans = 0
    for u in range(1, N + 1):
        ans += dist[u] + cycle_sz[u]

    print(ans)

if __name__ == "__main__":
    main()