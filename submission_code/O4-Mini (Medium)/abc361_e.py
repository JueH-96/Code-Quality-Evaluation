import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin
    # Read number of cities
    line = data.readline().strip()
    if not line:
        return
    n = int(line)
    # Build adjacency list
    adj = [[] for _ in range(n + 1)]
    total_weight = 0
    for _ in range(n - 1):
        a, b, c = map(int, data.readline().split())
        adj[a].append((b, c))
        adj[b].append((a, c))
        total_weight += c

    # Function to find farthest node and its distance from a start
    def find_farthest(start):
        dist = [-1] * (n + 1)
        dist[start] = 0
        stack = [start]
        parent = [-1] * (n + 1)
        far_node = start
        far_dist = 0

        while stack:
            u = stack.pop()
            d_u = dist[u]
            # Check if this is the farthest seen so far
            if d_u > far_dist:
                far_dist = d_u
                far_node = u
            # Traverse neighbors
            for v, w in adj[u]:
                if v == parent[u]:
                    continue
                parent[v] = u
                dist[v] = d_u + w
                stack.append(v)

        return far_node, far_dist

    # First pass: from node 1 (or any node) find one endpoint of diameter
    node_u, _ = find_farthest(1)
    # Second pass: from node_u find the diameter length
    _, diameter = find_farthest(node_u)

    # The minimal travel is 2 * sum_of_weights - diameter
    answer = 2 * total_weight - diameter
    print(answer)

if __name__ == "__main__":
    main()