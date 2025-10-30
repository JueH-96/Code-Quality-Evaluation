def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    
    adj = [[] for _ in range(N + 1)]
    for _ in range(N - 1):
        u = int(data[idx])
        v = int(data[idx + 1])
        adj[u].append(v)
        adj[v].append(u)
        idx += 2
    
    size = [1] * (N + 1)
    stack = [(1, -1, False)]
    
    while stack:
        u, p, visited = stack.pop()
        if not visited:
            stack.append((u, p, True))
            for v in adj[u]:
                if v != p:
                    stack.append((v, u, False))
        else:
            for v in adj[u]:
                if v != p:
                    size[u] += size[v]
    
    children = adj[1]
    if not children:
        print(1)
        return
    
    sum_children = sum(size[v] for v in children)
    max_child = max(size[v] for v in children)
    print(sum_children - max_child + 1)

if __name__ == "__main__":
    main()