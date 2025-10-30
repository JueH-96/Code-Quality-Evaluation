import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    edges = [[] for _ in range(N + 1)]
    degree = [0] * (N + 1)
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        edges[u].append(v)
        edges[v].append(u)
        degree[u] += 1
        degree[v] += 1

    max_size = 0
    for u in range(1, N + 1):
        s = [degree[v] - 1 for v in edges[u]]
        s.sort()
        k = len(s)
        current_max = 0
        for x in range(1, k + 1):
            y = s[-x]
            current_val = x * (y + 1) + 1
            if current_val > current_max:
                current_max = current_val
        if current_max > max_size:
            max_size = current_max

    print(N - max_size)

if __name__ == "__main__":
    main()