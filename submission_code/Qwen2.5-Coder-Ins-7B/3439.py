class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        from collections import defaultdict, deque
        
        def bfs(start, graph):
            visited = set()
            queue = deque([start])
            visited.add(start)
            level = -1
            while queue:
                level += 1
                for _ in range(len(queue)):
                    node = queue.popleft()
                    for neighbor in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
            return level
        
        def find_furthest_node(start, graph):
            max_level = -1
            furthest_node = -1
            visited = set()
            queue = deque([(start, 0)])
            visited.add(start)
            while queue:
                node, level = queue.popleft()
                if level > max_level:
                    max_level = level
                    furthest_node = node
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, level + 1))
            return furthest_node
        
        def get_diameter(graph):
            start = find_furthest_node(0, graph)
            return bfs(start, graph)
        
        graph1 = defaultdict(list)
        for u, v in edges1:
            graph1[u].append(v)
            graph1[v].append(u)
        
        graph2 = defaultdict(list)
        for u, v in edges2:
            graph2[u].append(v)
            graph2[v].append(u)
        
        diameter1 = get_diameter(graph1)
        diameter2 = get_diameter(graph2)
        
        min_diameter = float('inf')
        for u in range(len(edges1) + 1):
            for v in range(len(edges2) + 1):
                if u < len(edges1):
                    graph1[u].append(v)
                    graph1[v].append(u)
                if v < len(edges2):
                    graph2[v].append(u)
                    graph2[u].append(v)
                
                merged_diameter = max(get_diameter(graph1), get_diameter(graph2))
                min_diameter = min(min_diameter, merged_diameter)
                
                if u < len(edges1):
                    graph1[u].remove(v)
                    graph1[v].remove(u)
                if v < len(edges2):
                    graph2[v].remove(u)
                    graph2[u].remove(v)
        
        return min_diameter