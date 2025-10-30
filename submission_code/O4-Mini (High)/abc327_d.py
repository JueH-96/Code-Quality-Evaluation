import sys
from collections import deque

def main():
    input = sys.stdin.readline
    n, m = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # Build graph and check self-loops
    adj = [[] for _ in range(n+1)]
    for a, b in zip(A, B):
        if a == b:
            print("No")
            return
        adj[a].append(b)
        adj[b].append(a)

    # -1 = unvisited, 0/1 = two colors
    color = [-1] * (n+1)
    queue = deque()

    # Check each connected component for bipartiteness
    for i in range(1, n+1):
        if color[i] == -1 and adj[i]:
            color[i] = 0
            queue.append(i)
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        queue.append(v)
                    elif color[v] == color[u]:
                        print("No")
                        return

    print("Yes")

if __name__ == "__main__":
    main()