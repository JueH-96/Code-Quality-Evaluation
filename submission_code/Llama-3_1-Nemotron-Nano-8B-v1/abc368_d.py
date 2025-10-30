import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N, K = map(int, sys.stdin.readline().split())
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a, b = map(int, sys.stdin.readline().split())
        edges[a].append(b)
        edges[b].append(a)
    V = list(map(int, sys.stdin.readline().split()))
    is_v = [False] * (N + 1)
    for v in V:
        is_v[v] = True

    # Build children array using BFS
    root = 1
    parent = [0] * (N + 1)
    children = [[] for _ in range(N + 1)]
    q = deque([root])
    parent[root] = -1  # mark root's parent as -1 to avoid None
    while q:
        u = q.popleft()
        for v in edges[u]:
            if parent[u] != v:
                parent[v] = u
                children[u].append(v)
                q.append(v)

    # Post-order traversal to compute count
    count = [0] * (N + 1)
    stack = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if not visited:
            stack.append((node, True))
            # Push children in reverse to process in order
            for child in reversed(children[node]):
                stack.append((child, False))
        else:
            cnt = 0
            if is_v[node]:
                cnt += 1
            for child in children[node]:
                cnt += count[child]
            count[node] = cnt

    ans = 0
    for u in range(1, N + 1):
        if is_v[u]:
            ans += 1
        else:
            if count[u] >= 1 and (K - count[u]) >= 1:
                ans += 1
    print(ans)

if __name__ == '__main__':
    main()