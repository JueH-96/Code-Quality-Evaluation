import sys
import collections

def solve():
    """
    Reads graph input, calculates the solution, and prints it.
    """
    try:
        # Read the number of cities
        N = int(sys.stdin.readline())
    except (IOError, ValueError):
        # Handle cases with no or invalid input
        return

    # Edge case: If there is only one city, no travel is needed.
    if N == 1:
        print(0)
        return

    # Adjacency list to represent the tree.
    # adj[u] will store a list of tuples (v, w), where v is a neighbor of u and w is the edge weight.
    # We use 1-based indexing for cities 1 to N.
    adj = [[] for _ in range(N + 1)]
    total_weight = 0

    # Read the N-1 edges and build the graph.
    # At the same time, calculate the sum of all edge weights.
    for _ in range(N - 1):
        A, B, C = map(int, sys.stdin.readline().split())
        adj[A].append((B, C))
        adj[B].append((A, C))
        total_weight += C

    def bfs_farthest(start_node):
        """
        Performs BFS from a given start node to find the farthest node in the tree.
        This works correctly for trees with non-negative edge weights.
        Returns a tuple: (farthest_node, distance_to_farthest_node).
        """
        # dist[i] stores the shortest distance from start_node to node i.
        dist = [-1] * (N + 1)
        dist[start_node] = 0
        
        queue = collections.deque([start_node])

        while queue:
            u = queue.popleft()
            for v, weight in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + weight
                    queue.append(v)
        
        # After the BFS, iterate through all nodes to find the one with the maximum distance.
        max_dist = -1
        farthest_node = -1
        for i in range(1, N + 1):
            if dist[i] > max_dist:
                max_dist = dist[i]
                farthest_node = i
                
        return farthest_node, max_dist

    # Find the diameter of the tree using two BFS runs.
    # 1. Find an endpoint of a diameter by starting from an arbitrary node (e.g., 1).
    endpoint, _ = bfs_farthest(1)
    
    # 2. Find the diameter length by starting from that endpoint.
    _, diameter = bfs_farthest(endpoint)

    # The minimum tour length is the cost of traversing every edge twice,
    # minus the longest path (the diameter), as we don't need to travel back along this path.
    result = 2 * total_weight - diameter
    print(result)

solve()