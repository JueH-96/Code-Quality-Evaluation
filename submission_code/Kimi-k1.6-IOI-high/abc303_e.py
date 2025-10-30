def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    
    adj = [[] for _ in range(N + 1)]
    degree = [0] * (N + 1)
    
    for _ in range(N - 1):
        u = int(data[idx])
        v = int(data[idx + 1])
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
    
    centers = set()
    for u in range(1, N + 1):
        if degree[u] == 1:
            for v in adj[u]:
                centers.add(v)
    
    levels = [degree[v] for v in centers]
    levels.sort()
    print(' '.join(map(str, levels)))

if __name__ == "__main__":
    main()