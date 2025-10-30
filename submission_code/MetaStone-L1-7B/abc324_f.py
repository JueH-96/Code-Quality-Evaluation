import sys
from collections import defaultdict

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    edges = []
    edges_by_u = defaultdict(list)
    for _ in range(M):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        c = int(input[ptr])
        ptr += 1
        edges.append((u, v, b, c))
        edges_by_u[u].append((v, b, c))

    low = 0.0
    high = 1e18

    for _ in range(100):
        mid = (low + high) / 2
        sum_ = [-float('inf')] * (N + 1)
        sum_[1] = 0.0

        for u in range(1, N + 1):
            current = sum_[u]
            if current == -float('inf'):
                continue
            for (v, b, c) in edges_by_u.get(u, []):
                new_b = b - mid * c
                if sum_[v] < current + new_b:
                    sum_[v] = current + new_b

        if sum_[N] >= 0:
            low = mid
        else:
            high = mid

    print("{0:.15f}".format(low))

if __name__ == '__main__':
    main()