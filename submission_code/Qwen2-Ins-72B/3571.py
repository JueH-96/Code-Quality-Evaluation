class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        graph = {}
        for i, (x, y) in enumerate(coordinates):
            graph[(x, y)] = []
            for j in range(len(coordinates)):
                if i != j:
                    x2, y2 = coordinates[j]
                    if x < x2 and y < y2:
                        graph[(x, y)].append((x2, y2))
        
        @lru_cache(None)
        def dfs(point):
            x, y = point
            return 1 + max((dfs(next_point) for next_point in graph[point]), default=0)
        
        return dfs(tuple(coordinates[k]))