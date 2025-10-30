import sys
from collections import deque

input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1

    edges = []
    for _ in range(N - 1):
        A = int(data[index])
        B = int(data[index + 1])
        index += 2
        edges.append((A, B))

    C = list(map(int, data[index:index + N]))
    index += N

    tree = [[] for _ in range(N + 1)]
    for A, B in edges:
        tree[A].append(B)
        tree[B].append(A)

    def bfs(start):
        distance = [-1] * (N + 1)
        distance[start] = 0
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for neighbor in tree[current]:
                if distance[neighbor] == -1:
                    distance[neighbor] = distance[current] + 1
                    queue.append(neighbor)

        return distance

    min_f = float('inf')

    for v in range(1, N + 1):
        distance = bfs(v)
        f_v = sum(C[i - 1] * distance[i] for i in range(1, N + 1))
        min_f = min(min_f, f_v)

    print(min_f)

if __name__ == "__main__":
    solve()