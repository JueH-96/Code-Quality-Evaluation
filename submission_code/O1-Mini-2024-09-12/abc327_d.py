import sys
import sys
import sys
import sys
def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    from collections import deque

    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    adjacency = [[] for _ in range(N + 1)]
    for a, b in zip(A, B):
        if a == b:
            print("No")
            return
        adjacency[a].append(b)
        adjacency[b].append(a)

    color = [ -1 ] * (N + 1)

    for node in range(1, N + 1):
        if color[node] == -1:
            queue = deque()
            queue.append(node)
            color[node] = 0
            while queue:
                current = queue.popleft()
                for neighbor in adjacency[current]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[current]
                        queue.append(neighbor)
                    elif color[neighbor] == color[current]:
                        print("No")
                        return
    print("Yes")

if __name__ == "__main__":
    main()