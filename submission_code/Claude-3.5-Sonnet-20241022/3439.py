class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def build_graph(edges):
            graph = {}
            for u, v in edges:
                if u not in graph:
                    graph[u] = []
                if v not in graph:
                    graph[v] = []
                graph[u].append(v)
                graph[v].append(u)
            return graph
        
        def find_farthest(graph, start):
            visited = set()
            max_dist = 0
            farthest_node = start
            
            def dfs(node, dist):
                nonlocal max_dist, farthest_node
                visited.add(node)
                
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                    
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor, dist + 1)
                        
            dfs(start, 0)
            return farthest_node, max_dist
        
        def get_height(graph, node, parent):
            height = 0
            for neighbor in graph[node]:
                if neighbor != parent:
                    height = max(height, 1 + get_height(graph, neighbor, node))
            return height
        
        # Build adjacency lists for both trees
        graph1 = build_graph(edges1)
        graph2 = build_graph(edges2)
        
        # Find diameters of both trees
        end1, _ = find_farthest(graph1, 0)
        _, diam1 = find_farthest(graph1, end1)
        
        end2, _ = find_farthest(graph2, 0)
        _, diam2 = find_farthest(graph2, end2)
        
        # Try all possible connections between nodes
        min_diameter = float('inf')
        for node1 in graph1:
            height1 = get_height(graph1, node1, None)
            for node2 in graph2:
                height2 = get_height(graph2, node2, None)
                # New diameter will be maximum of:
                # 1. Original diameter of tree1
                # 2. Original diameter of tree2
                # 3. Height from node1 to leaf + 1 + Height from node2 to leaf
                curr_diameter = max(diam1, diam2, height1 + height2 + 1)
                min_diameter = min(min_diameter, curr_diameter)
                
        return min_diameter