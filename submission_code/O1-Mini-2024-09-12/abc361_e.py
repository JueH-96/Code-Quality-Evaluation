from collections import deque
import sys

def solve():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    adj = [[] for _ in range(N+1)]
    total = 0
    idx = 1
    for _ in range(N-1):
        A = int(data[idx])
        B = int(data[idx+1])
        C = int(data[idx+2])
        adj[A].append((B, C))
        adj[B].append((A, C))
        total += C
        idx +=3

    def bfs(start):
        visited = [False]*(N+1)
        q = deque()
        q.append((start, 0))
        visited[start]=True
        max_dist =0
        max_node = start
        while q:
            node, dist = q.popleft()
            if dist > max_dist:
                max_dist = dist
                max_node = node
            for neighbor, cost in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor]=True
                    q.append((neighbor, dist + cost))
        return max_node, max_dist

    far_node, _ = bfs(1)
    _, diameter = bfs(far_node)
    answer = 2 * total - diameter
    print(answer)