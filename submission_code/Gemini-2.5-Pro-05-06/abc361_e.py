import collections
import sys

# For faster I/O
input = sys.stdin.readline

def main():
    N = int(input())

    # Adjacency list: adj[i] stores list of (neighbor, weight) for node i
    # Using 1-based indexing for nodes to match problem statement (nodes 1 to N)
    adj = [[] for _ in range(N + 1)] # Size N+1 for nodes 1...N
    total_edge_weight = 0

    # Constraints state N-1 roads will be given.
    for _ in range(N - 1):
        u, v, c = map(int, input().split())
        adj[u].append((v, c))
        adj[v].append((u, c))
        total_edge_weight += c

    # BFS function to find the farthest node from a given start_node_val
    # and the distance to that farthest node.
    def bfs_for_diameter(start_node_val):
        # Queue stores tuples of (node, current_distance_from_start_node_val)
        q = collections.deque([(start_node_val, 0)])
        
        # visited_nodes set to keep track of visited nodes to ensure each node's
        # distance is computed once and correctly (shortest path in tree is unique path)
        visited_nodes = {start_node_val} 

        farthest_node_in_bfs = start_node_val # Initialize with start node
        max_distance_in_bfs = 0          # Initialize with distance 0 to start node itself

        while q:
            current_node, current_dist = q.popleft()

            # If the current_node is farther than any node found so far, update.
            if current_dist > max_distance_in_bfs:
                max_distance_in_bfs = current_dist
                farthest_node_in_bfs = current_node
            
            # Explore neighbors of current_node
            for neighbor_node, edge_weight in adj[current_node]:
                if neighbor_node not in visited_nodes:
                    visited_nodes.add(neighbor_node)
                    q.append((neighbor_node, current_dist + edge_weight))
        
        return farthest_node_in_bfs, max_distance_in_bfs

    # To find the diameter of the tree:
    # Step 1: Start BFS from an arbitrary node (e.g., node 1) to find a node
    # that is farthest from it. This node (call it 'u_endpoint') must be
    # one of the endpoints of some diameter of the tree.
    # Node 1 is guaranteed to exist since N >= 2 as per constraints.
    u_endpoint, _ = bfs_for_diameter(1) # The distance value from this BFS is not directly used.

    # Step 2: Start another BFS from 'u_endpoint'. The node farthest from 'u_endpoint'
    # will be the other endpoint of the diameter. The distance to this farthest node
    # is the length of the diameter.
    # The other endpoint node itself ('v_endpoint') is not strictly needed for the final calculation,
    # only the diameter's length.
    _, diameter_length = bfs_for_diameter(u_endpoint)
    
    # The problem asks for the minimum travel distance required to start from a city
    # and visit all cities at least once.
    # This is a known result for trees: 2 * (sum of all edge weights) - (length of tree diameter).
    min_travel_distance = 2 * total_edge_weight - diameter_length
    print(min_travel_distance)

if __name__ == '__main__':
    main()