def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    K = int(data[idx])
    idx += 1
    
    edges = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        a = int(data[idx])
        idx += 1
        b = int(data[idx])
        idx += 1
        edges[a].append(b)
        edges[b].append(a)
    
    if K == 0:
        print(0)
        return
    
    k_nodes = list(map(int, data[idx:idx+K]))
    idx += K
    k_set = set(k_nodes)
    k_total = K
    
    # Initialize parent and cnt arrays
    parent = [0] * (N + 1)
    cnt = [0] * (N + 1)
    
    # Using an iterative post-order traversal
    stack = []
    visited = [False] * (N + 1)
    stack.append((1, False, -1))  # (node, is_visited, parent)
    
    while stack:
        u, is_visited, p = stack.pop()
        if not is_visited:
            parent[u] = p
            # Push back as visited, then push children
            stack.append((u, True, p))
            # Push children in reverse order to process them in order
            for v in reversed(edges[u]):
                if v != p:
                    stack.append((v, False, u))
        else:
            if u in k_set:
                cnt[u] = 1
            else:
                cnt[u] = 0
            for v in edges[u]:
                if v != parent[u]:
                    cnt[u] += cnt[v]
    
    ans = 0
    for u in range(1, N + 1):
        if u in k_set:
            ans += 1
        else:
            if cnt[u] > 0 and (k_total - cnt[u]) > 0:
                ans += 1
    print(ans)

if __name__ == '__main__':
    main()