import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    adj = [[] for _ in range(N + 1)]
    sum_edges = 0
    for _ in range(N - 1):
        A, B, C = map(int, sys.stdin.readline().split())
        adj[A].append((B, C))
        adj[B].append((A, C))
        sum_edges += C

    def find_farthest(start):
        max_dist = 0
        far_node = start
        stack = [(start, -1, 0)]
        while stack:
            u, parent, dist = stack.pop()
            if dist > max_dist:
                max_dist = dist
                far_node = u
            for v, w in adj[u]:
                if v != parent:
                    stack.append((v, u, dist + w))
        return far_node, max_dist

    u, _ = find_farthest(1)
    v, diameter = find_farthest(u)
    print(2 * sum_edges - diameter)

if __name__ == '__main__':
    main()