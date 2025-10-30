import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1
    edges = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(input[idx]); idx += 1
        v = int(input[idx]); idx += 1
        b = int(input[idx]); idx += 1
        c = int(input[idx]); idx += 1
        edges[u].append((v, b, c))
    
    low = 0.0
    high = 1e14
    for _ in range(100):
        mid = (low + high) / 2
        dist = [-1e18] * (N + 1)
        dist[1] = 0.0
        for u in range(1, N + 1):
            for (v, b, c) in edges[u]:
                new_val = dist[u] + (b - mid * c)
                if new_val > dist[v]:
                    dist[v] = new_val
        if dist[N] >= -1e-9:
            low = mid
        else:
            high = mid
    print("{0:.15f}".format(low))

if __name__ == "__main__":
    main()