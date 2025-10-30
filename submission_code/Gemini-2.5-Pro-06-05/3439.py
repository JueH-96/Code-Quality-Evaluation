import collections
from typing import List

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        # The number of nodes in a tree is one more than the number of edges.
        n = len(edges1) + 1
        m = len(edges2) + 1
        
        # First, we need to find the properties of each tree independently.
        # The key properties are the diameter and radius.
        diam1 = self._get_diameter(n, edges1)
        diam2 = self._get_diameter(m, edges2)
        
        # The radius of a tree can be calculated from its diameter.
        # For any tree T, radius(T) = ceil(diameter(T) / 2).
        # In integer arithmetic, this is (diameter + 1) // 2.
        radius1 = (diam1 + 1) // 2
        radius2 = (diam2 + 1) // 2
        
        # The diameter of the merged tree is the maximum of:
        # 1. The diameter of the first tree.
        # 2. The diameter of the second tree.
        # 3. The length of the longest path that crosses the new edge.
        
        # To get the *minimum* possible diameter, we must choose the connection points
        # to minimize the length of the longest path crossing the new edge.
        # This is achieved by connecting the centers of the two trees.
        # The length of this path will be radius1 + 1 + radius2.
        
        cross_diameter = radius1 + radius2 + 1
        
        return max(diam1, diam2, cross_diameter)

    def _get_diameter(self, num_nodes: int, edges: List[List[int]]) -> int:
        """
        Calculates the diameter of a tree using a two-pass BFS algorithm.
        """
        if num_nodes <= 1:
            return 0
        
        adj = [[] for _ in range(num_nodes)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        def _bfs(start_node: int) -> tuple[int, int]:
            """
            Performs a BFS starting from a given node to find the farthest node
            and the distance to it.
            Returns: A tuple (farthest_node, distance).
            """
            distances = [-1] * num_nodes
            distances[start_node] = 0
            queue = collections.deque([start_node])
            
            # In a tree, the last node visited in a BFS from a single source
            # will be one of the nodes at the maximum distance.
            last_node = start_node
            
            while queue:
                u = queue.popleft()
                last_node = u
                
                for v in adj[u]:
                    if distances[v] == -1:
                        distances[v] = distances[u] + 1
                        queue.append(v)
            
            return last_node, distances[last_node]

        # The standard algorithm to find the diameter of a tree is as follows:
        # 1. Pick an arbitrary node `s` (e.g., node 0).
        # 2. Run BFS from `s` to find the node `u` that is farthest from `s`.
        #    This node `u` is guaranteed to be one of the endpoints of a diameter path.
        # 3. Run BFS from `u`. The farthest node `v` from `u` is the other
        #    endpoint of the diameter, and the distance d(u,v) is the diameter.
        
        # Step 1: Find one endpoint of a diameter path.
        endpoint1, _ = _bfs(0)
        
        # Step 2: Find the other endpoint and the diameter length.
        _, diameter = _bfs(endpoint1)
        
        return diameter