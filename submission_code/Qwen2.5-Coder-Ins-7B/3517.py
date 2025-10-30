class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        graph = {i: i + 1 for i in range(n - 1)}
        result = []
        current_min = n - 1
        
        for u, v in queries:
            if v not in graph:
                graph[v] = v
            if u in graph:
                current_min = min(current_min, graph[v] - graph[u] + 1)
            result.append(current_min)
        
        return result