# YOUR CODE HERE
import sys
import threading
import sys
def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    from collections import defaultdict, deque

    tree = defaultdict(list)
    degrees = [0] * (N + 1)
    for _ in range(N - 1):
        u, v = map(int, sys.stdin.readline().split())
        tree[u].append(v)
        tree[v].append(u)
        degrees[u] += 1
        degrees[v] += 1

    visited = [False] * (N + 1)
    answer = 0

    for node in range(1, N + 1):
        if degrees[node] == 3 and not visited[node]:
            # Start BFS/DFS from this node
            # Collect connected component of degree-3 nodes
            component = []
            stack = [node]
            visited[node] = True
            while stack:
                u = stack.pop()
                component.append(u)
                for v in tree[u]:
                    if not visited[v] and degrees[v] == 3:
                        visited[v] = True
                        stack.append(v)

            # Find boundary degree-2 nodes
            boundary = set()
            for u in component:
                for v in tree[u]:
                    if degrees[v] == 2:  # boundary node
                        boundary.add(v)

            k = len(boundary)
            if k >= 2:
                answer += k * (k - 1) // 2

    print(answer)

threading.Thread(target=main).start()