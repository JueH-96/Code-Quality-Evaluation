import sys

def main():
    n = int(sys.stdin.readline())
    adj = [[] for _ in range(n + 1)]
    total_sum = 0
    for _ in range(n - 1):
        a, b, c = map(int, sys.stdin.readline().split())
        adj[a].append((b, c))
        adj[b].append((a, c))
        total_sum += c

    def get_farthest(start):
        max_dist = 0
        farthest = start
        stack = [(start, -1, 0)]
        while stack:
            node, parent, dist = stack.pop()
            if dist > max_dist:
                max_dist = dist
                farthest = node
            for neighbor, weight in adj[node]:
                if neighbor != parent:
                    stack.append((neighbor, node, dist + weight))
        return farthest, max_dist

    u, _ = get_farthest(1)
    _, diameter = get_farthest(u)
    print(2 * total_sum - diameter)

main()