def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    edges = data[1:]
    
    adj = [[] for _ in range(N + 1)]
    degrees = [0] * (N + 1)
    
    for i in range(0, len(edges), 2):
        u = int(edges[i])
        v = int(edges[i + 1])
        adj[u].append(v)
        adj[v].append(u)
        degrees[u] += 1
        degrees[v] += 1
    
    max_total = 0
    for center in range(1, N + 1):
        neighbors = adj[center]
        degrees_of_neighbors = [degrees[neighbor] for neighbor in neighbors]
        degrees_of_neighbors.sort()
        k = len(neighbors)
        for x in range(1, k + 1):
            d_x = degrees_of_neighbors[x - 1]
            if d_x >= 2:
                y = d_x - 1
                total = 1 + x + x * y
                if total > max_total:
                    max_total = total
    deletions = N - max_total
    print(deletions)

if __name__ == "__main__":
    main()