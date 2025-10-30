from collections import defaultdict

class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        def dfs(node, parent, graph, dist):
            for neighbor in graph[node]:
                if neighbor != parent:
                    dist[neighbor] = dist[node] + 1
                    dfs(neighbor, node, graph, dist)
        
        def get_diameter(graph):
            start = 0
            dist = [0] * len(graph)
            dfs(start, -1, graph, dist)
            farthest = dist.index(max(dist))
            dist = [0] * len(graph)
            dfs(farthest, -1, graph, dist)
            return max(dist)
        
        n, m = len(edges1) + 1, len(edges2) + 1
        graph1, graph2 = defaultdict(list), defaultdict(list)
        
        for a, b in edges1:
            graph1[a].append(b)
            graph1[b].append(a)
        
        for u, v in edges2:
            graph2[u].append(v)
            graph2[v].append(u)
        
        min_diameter = float('inf')
        
        for i in range(n):
            for j in range(m):
                graph1[i].append(j + n)
                graph1[j + n].append(i)
                graph2[i].append(j + n)
                graph2[j + n].append(i)
                
                diameter = max(get_diameter(graph1), get_diameter(graph2))
                min_diameter = min(min_diameter, diameter)
                
                graph1[i].remove(j + n)
                graph1[j + n].remove(i)
                graph2[i].remove(j + n)
                graph2[j + n].remove(i)
        
        return min_diameter