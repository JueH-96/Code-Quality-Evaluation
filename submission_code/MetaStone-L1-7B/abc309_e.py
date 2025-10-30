from collections import deque

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    # Build the tree
    p = [0] * (N + 1)  # p[2] is the parent of 2, etc.
    for i in range(2, N + 1):
        p[i] = int(input[ptr])
        ptr += 1

    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for i in range(2, N + 1):
        parent = p[i]
        adj[parent].append(i)

    # Compute depth for each node
    depth = [0] * (N + 1)
    q = deque()
    q.append(1)
    visited = [False] * (N + 1)
    visited[1] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                depth[v] = depth[u] + 1
                q.append(v)

    # Read insurance data
    max_y = [0] * (N + 1)
    for _ in range(M):
        x = int(input[ptr])
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        if y > max_y[x]:
            max_y[x] = y

    # BFS to compute covered nodes
    covered = [False] * (N + 1)
    q = deque()
    q.append((1, depth[1] + max_y[1]))
    covered[1] = True

    while q:
        u, current_max = q.popleft()
        for v in adj[u]:
            child_max = max(current_max, depth[u] + max_y[v])
            if child_max >= depth[v] and not covered[v]:
                covered[v] = True
                q.append((v, child_max))

    # Count covered nodes
    count = sum(covered)
    print(count)

if __name__ == '__main__':
    main()