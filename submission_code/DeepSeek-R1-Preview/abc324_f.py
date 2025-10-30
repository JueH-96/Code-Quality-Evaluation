import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    edges = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(input[idx])
        idx += 1
        v = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        c = int(input[idx])
        idx += 1
        edges[u].append((v, b, c))

    low = 0.0
    high = 1e18
    for _ in range(100):
        mid = (low + high) / 2
        max_sum = [-float('inf')] * (N + 1)
        max_sum[1] = 0.0
        for u in range(1, N + 1):
            if max_sum[u] == -float('inf'):
                continue
            for (v, b, c) in edges[u]:
                new_sum = max_sum[u] + (b - mid * c)
                if new_sum > max_sum[v]:
                    max_sum[v] = new_sum
        if max_sum[N] >= 0:
            low = mid
        else:
            high = mid

    print("{0:.15f}".format(low))

if __name__ == '__main__':
    main()