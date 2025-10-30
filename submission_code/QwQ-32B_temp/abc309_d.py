import sys
from collections import deque

def main():
    N1, N2, M = map(int, sys.stdin.readline().split())
    total_nodes = N1 + N2
    adj = [[] for _ in range(total_nodes + 1)]  # 1-based indexing

    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)

    # Compute distances from node 1 (first component)
    dist1 = [-1] * (total_nodes + 1)
    q = deque()
    q.append(1)
    dist1[1] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist1[v] == -1:
                dist1[v] = dist1[u] + 1
                q.append(v)

    max_d1 = max(dist1[i] for i in range(1, N1 + 1))

    # Compute distances from the end node (second component)
    end_node = total_nodes
    dist2 = [-1] * (total_nodes + 1)
    q = deque()
    q.append(end_node)
    dist2[end_node] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist2[v] == -1:
                dist2[v] = dist2[u] + 1
                q.append(v)

    max_d2 = 0
    for i in range(N1 + 1, total_nodes + 1):
        if dist2[i] > max_d2:
            max_d2 = dist2[i]

    ans = max_d1 + max_d2 + 1
    print(ans)

if __name__ == "__main__":
    main()