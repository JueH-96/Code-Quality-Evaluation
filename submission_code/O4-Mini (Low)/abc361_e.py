import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input().strip())
    adj = [[] for _ in range(N+1)]
    total_weight = 0

    for _ in range(N-1):
        a, b, c = map(int, input().split())
        adj[a].append((b, c))
        adj[b].append((a, c))
        total_weight += c

    # Helper to find farthest node and distance from a start
    def farthest(start):
        visited = [False] * (N+1)
        stack = [(start, 0)]
        visited[start] = True
        far_node = start
        far_dist = 0
        while stack:
            node, dist = stack.pop()
            if dist > far_dist:
                far_dist = dist
                far_node = node
            for (nxt, w) in adj[node]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append((nxt, dist + w))
        return far_node, far_dist

    # First pass: from node 1 (or any), find one end of diameter
    u, _ = farthest(1)
    # Second pass: from u, find the actual diameter length
    v, diameter = farthest(u)

    # Minimum travel = twice all edges minus the longest path
    answer = 2 * total_weight - diameter
    print(answer)

if __name__ == "__main__":
    main()