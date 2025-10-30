import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    degree = [0] * (N + 1)
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        degree[u] += 1
        degree[v] += 1

    candidates = []
    for node in range(1, N + 1):
        if degree[node] >= 2:
            candidates.append(node)

    # Sort candidates by (degree + 1) in ascending order
    candidates.sort(key=lambda x: (degree[x] + 1))

    total_sum = sum(degree[node] + 1 for node in candidates)
    delta = total_sum - N

    exclude_count = 0
    current_exclude = 0
    for i in range(len(candidates)):
        current_node = candidates[i]
        val = degree[current_node] + 1
        if current_exclude + val > delta:
            break
        current_exclude += val
        exclude_count += 1

    # The remaining nodes form the subset S
    L = [degree[node] for node in candidates[exclude_count:]]
    L.sort()
    print(' '.join(map(str, L)))

if __name__ == "__main__":
    main()