class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        # Create a graph to represent character transformations
        graph = {}
        for i in range(len(original)):
            if original[i] not in graph:
                graph[original[i]] = {}
            graph[original[i]][changed[i]] = min(graph[original[i]].get(changed[i], float('inf')), cost[i])

        # Floyd-Warshall algorithm to find shortest paths
        for k in 'abcdefghijklmnopqrstuvwxyz':
            for i in 'abcdefghijklmnopqrstuvwxyz':
                for j in 'abcdefghijklmnopqrstuvwxyz':
                    if k in graph and i in graph and j in graph[k]:
                        if i not in graph:
                            graph[i] = {}
                        graph[i][j] = min(graph[i].get(j, float('inf')), 
                                          graph[i].get(k, float('inf')) + graph[k][j])

        # Calculate the minimum cost to transform source to target
        total_cost = 0
        for s, t in zip(source, target):
            if s != t:
                if s not in graph or t not in graph[s]:
                    return -1
                total_cost += graph[s][t]

        return total_cost