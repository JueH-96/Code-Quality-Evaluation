import sys

def main():
    sys.setrecursionlimit(1 << 25)
    n, k = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        a, b = map(int, sys.stdin.readline().split())
        adj[a].append(b)
        adj[b].append(a)
    v_list = list(map(int, sys.stdin.readline().split()))
    required = [False] * (n + 1)
    for v in v_list:
        required[v] = True

    parent = [0] * (n + 1)
    count = [0] * (n + 1)
    stack = [(1, False)]
    while stack:
        node, visited = stack.pop()
        if not visited:
            stack.append((node, True))
            for neighbor in reversed(adj[node]):
                if neighbor != parent[node]:
                    parent[neighbor] = node
                    stack.append((neighbor, False))
        else:
            cnt = 0
            if required[node]:
                cnt += 1
            for neighbor in adj[node]:
                if neighbor != parent[node]:
                    cnt += count[neighbor]
            count[node] = cnt

    edge_count = 0
    for v in range(2, n + 1):
        if count[v] >= 1 and (k - count[v]) >= 1:
            edge_count += 1
    print(edge_count + 1)

if __name__ == "__main__":
    main()