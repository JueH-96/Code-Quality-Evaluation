def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = data[1:]
    
    adj = [[] for _ in range(N + 1)]
    for i in range(0, len(edges), 2):
        u = int(edges[i])
        v = int(edges[i + 1])
        adj[u].append(v)
        adj[v].append(u)
    
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = len(adj[i])
    
    result = []
    for i in range(1, N + 1):
        if degrees[i] >= 2:
            for neighbor in adj[i]:
                if degrees[neighbor] == 1:
                    result.append(degrees[i])
                    break
    
    result.sort()
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()