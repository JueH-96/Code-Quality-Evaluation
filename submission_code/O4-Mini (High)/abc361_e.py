import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    adj = [[] for _ in range(N)]
    total_weight = 0
    for _ in range(N - 1):
        a, b, c = map(int, input().split())
        a -= 1
        b -= 1
        adj[a].append((b, c))
        adj[b].append((a, c))
        total_weight += c

    def farthest(start):
        visited = [False] * N
        stack = [(start, 0)]
        visited[start] = True
        far_node = start
        far_dist = 0
        while stack:
            u, dist = stack.pop()
            if dist > far_dist:
                far_dist = dist
                far_node = u
            for v, w in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append((v, dist + w))
        return far_node, far_dist

    # First find one endpoint of the diameter
    u, _ = farthest(0)
    # Then find the actual diameter length from that endpoint
    _, diameter = farthest(u)

    # Minimum traversal = 2 * sum_of_weights - diameter
    answer = 2 * total_weight - diameter
    print(answer)

if __name__ == "__main__":
    main()