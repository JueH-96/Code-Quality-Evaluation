import sys
from collections import defaultdict

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    edges = [[] for _ in range(N+1)]
    degrees = [0] * (N+1)
    for _ in range(N-1):
        u = int(input[idx])
        v = int(input[idx+1])
        idx += 2
        edges[u].append(v)
        edges[v].append(u)
        degrees[u] += 1
        degrees[v] += 1

    leaves_count = sum(1 for degree in degrees if degree == 1)
    C = (N + 2 - leaves_count) // 3

    list_2 = []
    list_rest = []
    for node in range(1, N+1):
        if degrees[node] == 2:
            list_2.append(node)
        elif degrees[node] > 2:
            list_rest.append(node)

    non_central_count = 2 * (C - 1)
    central_2_count = len(list_2) - non_central_count

    central_degrees = []
    for node in list_rest:
        central_degrees.append(degrees[node])
    central_degrees += [2] * central_2_count

    central_degrees.sort()
    print(' '.join(map(str, central_degrees)))

if __name__ == '__main__':
    main()