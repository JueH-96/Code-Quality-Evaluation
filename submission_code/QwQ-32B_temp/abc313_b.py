import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    reversed_adj = [[] for _ in range(N + 1)]  # Using 1-based indexing

    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        reversed_adj[B].append(A)  # Add reversed edge B -> A

    candidates = []
    for X in range(1, N + 1):
        visited = [False] * (N + 1)
        q = deque()
        q.append(X)
        visited[X] = True

        while q:
            node = q.popleft()
            for neighbor in reversed_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    q.append(neighbor)

        # Calculate the number of reachable nodes excluding X
        count = sum(visited[1:N+1]) - 1
        if count == 0:
            candidates.append(X)

    if len(candidates) == 1:
        print(candidates[0])
    else:
        print(-1)

if __name__ == "__main__":
    main()