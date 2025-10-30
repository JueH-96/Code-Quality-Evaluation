import collections
from typing import List

class Solution:
    def _get_diameter_of_tree(self, num_nodes: int, edges: List[List[int]]) -> int:
        # If there's only one node or no nodes, diameter is 0.
        # Problem constraints state num_nodes >= 1.
        if num_nodes <= 1:
            return 0
        
        adj = [[] for _ in range(num_nodes)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS utility function
        # Input: start_node for BFS for the current graph (defined by num_nodes and adj).
        # Returns: (farthest_node_from_start, distance_to_farthest_node)
        def bfs(start_node: int):
            # Initialize distances to -1 (infinity/unvisited)
            dist = [-1] * num_nodes
            dist[start_node] = 0
            
            # Queue for BFS: stores tuples of (node, current_distance_from_start_node)
            q = collections.deque([(start_node, 0)])
            
            # Track the farthest node found so far in this BFS traversal and its distance
            farthest_node_in_bfs = start_node
            max_d_in_bfs = 0
            
            while q:
                u, d = q.popleft()
                
                # If current node's distance d is farther than max_d_in_bfs, update
                if d > max_d_in_bfs:
                    max_d_in_bfs = d
                    farthest_node_in_bfs = u
                
                # Explore neighbors of u
                for v_neighbor in adj[u]:
                    if dist[v_neighbor] == -1: # If neighbor not visited
                        dist[v_neighbor] = d + 1
                        q.append((v_neighbor, d + 1))
            
            return farthest_node_in_bfs, max_d_in_bfs

        # Standard algorithm for finding the diameter of a tree:
        # 1. Start BFS from an arbitrary node (e.g., node 0) to find the farthest node from it.
        #    This farthest node (node1_endpoint) must be one endpoint of some diameter of the tree.
        node1_endpoint, _ = bfs(0) 
        
        # 2. Start BFS from node1_endpoint to find the farthest node from it.
        #    The distance to this new farthest node is the actual diameter of the tree.
        _, diameter_val = bfs(node1_endpoint)
        
        return diameter_val

    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        # Determine the number of nodes in each tree.
        # For a tree with N nodes, there are N-1 edges. So N = len(edges) + 1.
        n1 = len(edges1) + 1
        n2 = len(edges2) + 1
        
        # Calculate the diameter of the first tree (T1).
        d1 = self._get_diameter_of_tree(n1, edges1)
        # Calculate the diameter of the second tree (T2).
        d2 = self._get_diameter_of_tree(n2, edges2)
        
        # When connecting T1 and T2 with an edge (x,y) where x is in T1 and y is in T2,
        # new paths are formed that go through this edge.
        # A path u-...-x---y-...-v has length dist(u,x) + 1 + dist(y,v).
        # To minimize the maximum such length, we should pick x and y such that
        # max_u dist(u,x) and max_v dist(y,v) are minimized.
        # These x and y are the "centers" of their respective trees.
        # The distance from a center to its farthest node is ceil(Diameter/2).
        
        # Smallest possible maximum distance from x to any node in T1 (eccentricity of a center in T1).
        # This is ceil(d1/2). In integer arithmetic, this is (d1 + 1) // 2.
        min_ecc_in_t1 = (d1 + 1) // 2
        
        # Smallest possible maximum distance from y to any node in T2 (eccentricity of a center in T2).
        min_ecc_in_t2 = (d2 + 1) // 2
        
        # The minimum length of the "longest path that uses the new edge" is
        # min_ecc_in_t1 + 1 (for the new edge) + min_ecc_in_t2.
        min_len_of_path_through_new_edge = min_ecc_in_t1 + min_ecc_in_t2 + 1
        
        # The diameter of the merged tree will be the maximum of three values:
        # 1. The original diameter of T1 (paths entirely within T1).
        # 2. The original diameter of T2 (paths entirely within T2).
        # 3. The minimized length of the longest path that passes through the new connecting edge.
        
        # The question asks for the minimum possible diameter of the resulting tree.
        # By choosing centers to connect, we have minimized the third term. D1 and D2 are fixed.
        # So this construction gives the minimum possible diameter.
        ans = max(d1, d2, min_len_of_path_through_new_edge)
        
        return ans