import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1

    original = [[] for _ in range(N + 1)]
    reversed_adj = [[] for _ in range(N + 1)]
    for _ in range(M):
        a = int(input[idx])
        idx += 1
        b = int(input[idx])
        idx += 1
        original[a].append(b)
        reversed_adj[b].append(a)

    # Compute distance from 1 in the original graph
    distance = [-1] * (N + 1)
    distance[1] = 0
    q = deque([1])
    while q:
        u = q.popleft()
        for v in original[u]:
            if distance[v] == -1:
                distance[v] = distance[u] + 1
                q.append(v)

    # Compute reverse_distance from 1 in the reversed graph
    reverse_distance = [-1] * (N + 1)
    reverse_distance[1] = 0
    q = deque([1])
    while q:
        u = q.popleft()
        for v in reversed_adj[u]:
            if reverse_distance[v] == -1:
                reverse_distance[v] = reverse_distance[u] + 1
                q.append(v)

    min_cycle = float('inf')
    for u in range(1, N + 1):
        for v in original[u]:
            if distance[u] != -1 and reverse_distance[v] != -1:
                cycle_length = distance[u] + 1 + reverse_distance[v]
                if cycle_length < min_cycle:
                    min_cycle = cycle_length

    print(min_cycle if min_cycle != float('inf') else -1)

if __name__ == '__main__':
    main()