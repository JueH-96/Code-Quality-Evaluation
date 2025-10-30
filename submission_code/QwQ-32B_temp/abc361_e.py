import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N + 1)]  # 1-based indexing
    sum_edges = 0
    for _ in range(N - 1):
        A, B, C = map(int, sys.stdin.readline().split())
        adj[A].append((B, C))
        adj[B].append((A, C))
        sum_edges += C

    def get_farthest(start):
        visited = [False] * (N + 1)
        q = deque()
        q.append((start, 0))
        visited[start] = True
        max_dist = 0
        far_node = start
        while q:
            node, dist = q.popleft()
            for neighbor, cost in adj[node]:
                if not visited[neighbor]:
                    new_dist = dist + cost
                    if new_dist > max_dist:
                        max_dist = new_dist
                        far_node = neighbor
                    visited[neighbor] = True
                    q.append((neighbor, new_dist))
        return far_node, max_dist

    # First find one end of the diameter
    u, _ = get_farthest(1)
    # Now find the other end and the diameter
    v, diameter = get_farthest(u)
    print(2 * sum_edges - diameter)

if __name__ == "__main__":
    main()