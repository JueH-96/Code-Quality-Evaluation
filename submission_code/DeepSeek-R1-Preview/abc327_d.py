import sys
from collections import deque

def main():
    n, m = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Check if any A[i] equals B[i]
    for i in range(m):
        if A[i] == B[i]:
            print("No")
            return

    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    for i in range(m):
        a = A[i]
        b = B[i]
        adj[a].append(b)
        adj[b].append(a)

    colors = [-1] * (n + 1)

    for i in range(1, n + 1):
        if colors[i] == -1:
            queue = deque()
            queue.append(i)
            colors[i] = 0
            while queue:
                current = queue.popleft()
                for neighbor in adj[current]:
                    if colors[neighbor] == -1:
                        colors[neighbor] = 1 - colors[current]
                        queue.append(neighbor)
                    else:
                        if colors[neighbor] == colors[current]:
                            print("No")
                            return
    print("Yes")

if __name__ == "__main__":
    main()