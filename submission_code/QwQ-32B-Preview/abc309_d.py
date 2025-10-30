import sys
from collections import deque

def main():
    import sys
    from collections import deque

    # Read input
    N1, N2, M = map(int, sys.stdin.readline().split())
    edges = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]

    # Build adjacency lists for group1 and group2
    adj_group1 = [[] for _ in range(N1 + 1)]
    adj_group2 = [[] for _ in range(N2 + 1)]
    for a, b in edges:
        if 1 <= a <= N1 and 1 <= b <= N1:
            adj_group1[a].append(b)
            adj_group1[b].append(a)
        elif N1 < a <= N1 + N2 and N1 < b <= N1 + N2:
            adj_group2[a - N1].append(b - N1)
            adj_group2[b - N1].append(a - N1)
        else:
            # Ignore edges between group1 and group2 as they are not present initially
            pass

    # BFS from vertex 1 in group1
    dist_from_1 = [-1] * (N1 + 1)
    dist_from_1[1] = 0
    queue = deque([1])
    while queue:
        u = queue.popleft()
        for v in adj_group1[u]:
            if dist_from_1[v] == -1:
                dist_from_1[v] = dist_from_1[u] + 1
                queue.append(v)

    # BFS from vertex (N1 + N2) in group2
    dist_to_N1_N2 = [-1] * (N2 + 1)
    dist_to_N1_N2[N2] = 0  # vertex (N1 + N2) is vertex N2 in adj_group2
    queue = deque([N2])
    while queue:
        v = queue.popleft()
        for w in adj_group2[v]:
            if dist_to_N1_N2[w] == -1:
                dist_to_N1_N2[w] = dist_to_N1_N2[v] + 1
                queue.append(w)

    # Find maximum distances
    max_dist_from_1 = max(dist_from_1[1:N1 + 1])
    max_dist_to_N1_N2 = max(dist_to_N1_N2[1:N2 + 1])

    # Calculate maximum d
    d = max_dist_from_1 + 1 + max_dist_to_N1_N2
    print(d)

if __name__ == '__main__':
    main()