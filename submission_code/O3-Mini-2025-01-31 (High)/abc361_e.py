def main():
    import sys
    sys.setrecursionlimit(300000)
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    
    # Build the graph as an adjacency list (1-indexed)
    graph = [[] for _ in range(n+1)]
    total_length = 0
    for _ in range(n-1):
        u = int(next(it))
        v = int(next(it))
        w = int(next(it))
        graph[u].append((v, w))
        graph[v].append((u, w))
        total_length += w

    # Helper function: from a start city, find the farthest city and its distance using DFS (iterative)
    def get_farthest(start):
        farthest_distance = 0
        farthest_node = start
        # (current city, parent city, accumulated distance)
        stack = [(start, -1, 0)]
        while stack:
            node, parent, dist = stack.pop()
            if dist > farthest_distance:
                farthest_distance = dist
                farthest_node = node
            for nxt, w in graph[node]:
                if nxt == parent:
                    continue
                stack.append((nxt, node, dist + w))
        return farthest_distance, farthest_node

    # Finding the diameter of the tree:
    # 1. Start from an arbitrary city (city 1) to find one end of the longest path.
    _, far_node = get_farthest(1)
    # 2. From that farthest city, compute the longest distance (the diameter).
    diameter, _ = get_farthest(far_node)
    
    # The minimal travel distance is the total route traversing each road twice minus the longest single path.
    result = 2 * total_length - diameter
    sys.stdout.write(str(result))
    
if __name__ == "__main__":
    main()