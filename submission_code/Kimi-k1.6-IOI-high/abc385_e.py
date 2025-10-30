import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    adj = [[] for _ in range(N + 1)]
    degree = [0] * (N + 1)
    for _ in range(N - 1):
        u = int(input[idx])
        v = int(input[idx + 1])
        idx += 2
        adj[u].append(v)
        adj[v].append(u)
        degree[u] += 1
        degree[v] += 1
    
    max_total = 0
    for u in range(1, N + 1):
        neighbors = adj[u]
        values = [degree[v] - 1 for v in neighbors]
        values.sort(reverse=True)
        for x in range(1, len(values) + 1):
            y = values[x - 1]
            total = 1 + x + x * y
            if total > max_total:
                max_total = total
    print(N - max_total)

if __name__ == '__main__':
    main()