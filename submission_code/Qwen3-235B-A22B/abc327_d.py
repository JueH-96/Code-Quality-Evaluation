import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+M]))
    ptr += M
    B = list(map(int, input[ptr:ptr+M]))
    ptr += M

    # Check if any A_i == B_i
    for i in range(M):
        if A[i] == B[i]:
            print("No")
            return

    # Build adjacency list
    adj = [[] for _ in range(N + 1)]
    for i in range(M):
        u = A[i]
        v = B[i]
        adj[u].append(v)
        adj[v].append(u)

    # Check bipartition
    color = [-1] * (N + 1)
    visited = [False] * (N + 1)

    for u in range(1, N + 1):
        if not visited[u]:
            queue = deque()
            queue.append(u)
            visited[u] = True
            color[u] = 0
            while queue:
                current = queue.popleft()
                for neighbor in adj[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        color[neighbor] = 1 - color[current]
                        queue.append(neighbor)
                    else:
                        if color[neighbor] == color[current]:
                            print("No")
                            return

    print("Yes")

if __name__ == "__main__":
    main()