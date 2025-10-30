from collections import defaultdict
import heapq

class Solution:
    def maximizeSumOfWeights(self, edges, k):
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((-w, v))
            graph[v].append((-w, u))
        
        def dfs(node, parent):
            max_weight = [0] * (k+1)
            for weight, child in graph[node]:
                if child == parent:
                    continue
                child_weights = dfs(child, node)
                for i in range(k):
                    for j in range(k - i):
                        max_weight[i + j + 1] = max(max_weight[i + j + 1],
                                                   max_weight[i] + child_weights[j] - (weight,))
            
            for i in range(1, k + 1):
                max_weight[i] += weight
            return max_weight
        
        max_sum = dfs(0, None)
        return -max_sum[-1]