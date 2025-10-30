import collections

class Solution:
    def _bfs(self, n: int, adj: list[list[int]], start_node: int) -> tuple[list[int], int, int]:
        """
        Performs a Breadth-First Search (BFS) from a start_node in a tree.
        
        Args:
            n: Number of nodes in the tree.
            adj: Adjacency list representation of the tree.
            start_node: The node to start the BFS from.
            
        Returns:
            A tuple containing:
            - distances: A list where distances[i] is the shortest distance from start_node to node i.
            - farthest_node: The node farthest from start_node.
            - max_distance: The maximum distance found from start_node.
        """
        distances = [-1] * n
        q = collections.deque([(start_node, 0)])
        distances[start_node] = 0
        
        farthest_node = start_node
        max_dist = 0
        
        while q:
            curr, dist = q.popleft()
            
            # Update farthest node and max distance found so far
            if dist > max_dist:
                max_dist = dist
                farthest_node = curr
            
            for neighbor in adj[curr]:
                if distances[neighbor] == -1: # If not visited
                    distances[neighbor] = dist + 1
                    q.append((neighbor, dist + 1))
        
        return distances, farthest_node, max_dist

    def _get_tree_props(self, n: int, edges: list[list[int]]) -> tuple[int, int]:
        """
        Calculates the diameter and radius of a tree using three BFS traversals.
        
        Args:
            n: Number of nodes in the tree.
            edges: List of edges in the tree.
            
        Returns:
            A tuple containing:
            - diameter: The diameter of the tree.
            - radius: The radius of the tree.
        """
        if n == 1:
            return 0, 0 # A single node tree has diameter 0 and radius 0.

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Step 1: Perform BFS from an arbitrary node (e.g., node 0) to find one endpoint (u) of a diameter.
        # We only need the farthest_node from this BFS result for the next step.
        _, u, _ = self._bfs(n, adj, 0) 
        
        # Step 2: Perform BFS from 'u' to find the actual diameter and its other endpoint (v).
        # The 'max_distance' returned by this BFS is the tree's diameter.
        du, v, diameter = self._bfs(n, adj, u)
        
        # Step 3: Perform BFS from 'v' to get distances from 'v' to all other nodes.
        dv, _, _ = self._bfs(n, adj, v)
        
        # Step 4: Calculate eccentricities for all nodes.
        # For any node 'i', its eccentricity is the maximum distance from 'i' to any other node.
        # A known property of trees states that for any node 'i', its eccentricity is 
        # max(dist(i, u), dist(i, v)), where 'u' and 'v' are endpoints of *a* diameter.
        eccentricities = [0] * n
        for i in range(n):
            eccentricities[i] = max(du[i], dv[i])
        
        # Step 5: The radius of the tree is the minimum eccentricity among all nodes.
        radius = min(eccentricities)
        
        return diameter, radius

    def minimumDiameterAfterMerge(self, edges1: list[list[int]], edges2: list[list[int]]) -> int:
        n = len(edges1) + 1
        m = len(edges2) + 1

        # Calculate diameter and radius for the first tree (T1)
        diam1, rad1 = self._get_tree_props(n, edges1)
        
        # Calculate diameter and radius for the second tree (T2)
        diam2, rad2 = self._get_tree_props(m, edges2)
        
        # The diameter of the merged tree can be determined by considering three cases:
        # 1. The longest path is entirely within T1 (length = diam1).
        # 2. The longest path is entirely within T2 (length = diam2).
        # 3. The longest path crosses the new connecting edge.
        #    To minimize this crossing path, we connect a center node from T1 
        #    (node with minimum eccentricity, which is rad1) to a center node from T2 
        #    (node with minimum eccentricity, which is rad2).
        #    The length of such a path would be rad1 + 1 (for the connecting edge) + rad2.
        max_crossing_path = rad1 + rad2 + 1
        
        # The minimum possible diameter of the resulting tree is the maximum of these three possibilities.
        return max(diam1, diam2, max_crossing_path)