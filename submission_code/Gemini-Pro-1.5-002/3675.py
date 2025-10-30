class Solution:
    def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
        n = len(edges) + 1
        graph = [[] for _ in range(n)]
        for u, v, w in edges:
            graph[u].append((v, w))
            graph[v].append((u, w))

        max_sum = 0

        def dfs(u, parent, k):
            weights = []
            for v, w in graph[u]:
                if v != parent:
                    weights.extend(dfs(v, u, k))
            
            weights.sort(reverse=True)
            
            current_sum = 0
            count = 0
            for weight in weights:
                if count < k -1 :
                    current_sum += weight
                    count += 1
                
            
            
            if parent != -1:
                return [current_sum + w for v,w in graph[u] if v == parent]
            else:
                
                
                count = 0
                for weight in weights:
                    if count < k:
                        current_sum += weight
                        count += 1
                return [current_sum]
                
        max_sum = dfs(0, -1, k)[0]
        return max_sum