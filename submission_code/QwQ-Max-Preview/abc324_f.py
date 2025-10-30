import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    edges = [[] for _ in range(N + 1)]
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        c = int(input[ptr])
        ptr += 1
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
                delta = b - mid * c
                new_sum = max_sum[u] + delta
                if new_sum > max_sum[v]:
                    max_sum[v] = new_sum

        if max_sum[N] >= 0:
            low = mid
        else:
            high = mid

    print("{0:.16f}".format(low))

if __name__ == '__main__':
    main()