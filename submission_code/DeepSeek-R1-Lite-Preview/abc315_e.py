import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    index = 1
    adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        C_i = int(N_and_rest[index])
        index += 1
        prerequisites = []
        for _ in range(C_i):
            p = int(N_and_rest[index])
            prerequisites.append(p)
            index += 1
        adj[i] = prerequisites

    # Build reverse graph
    reverse_adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for p in adj[i]:
            reverse_adj[p].append(i)

    # BFS from book 1 in reverse graph to find S
    S = set()
    queue = deque([1])
    S.add(1)
    while queue:
        u = queue.popleft()
        for v in reverse_adj[u]:
            if v not in S:
                S.add(v)
                queue.append(v)

    # Build in-degree for Kahn's algorithm
    in_degree = [0] * (N + 1)
    for i in S:
        for p in adj[i]:
            if p in S:
                in_degree[i] += 1

    # Kahn's algorithm
    queue = deque()
    for i in S:
        if in_degree[i] == 0 and i != 1:
            queue.append(i)

    result = []
    while queue:
        u = queue.popleft()
        result.append(u)
        for v in reverse_adj[u]:
            if v in S:
                in_degree[v] -= 1
                if in_degree[v] == 0 and v != 1:
                    queue.append(v)

    # Exclude book 1 and print the result
    print(' '.join(map(str, result)))

if __name__ == '__main__':
    main()