# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    from collections import defaultdict, deque

    graph = defaultdict(list)
    for i in range(M):
        u = A[i]
        v = B[i]
        graph[u].append(v)
        graph[v].append(u)

    color = {}
    def bfs(start):
        queue = deque()
        queue.append(start)
        color[start] = 0
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v in color:
                    if color[v] == color[u]:
                        return False
                else:
                    color[v] = 1 - color[u]
                    queue.append(v)
        return True

    for node in range(1, N+1):
        if node in graph and node not in color:
            if not bfs(node):
                print("No")
                return
    print("Yes")

threading.Thread(target=main).start()