class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n = len(source)
        adj = {}
        for i in range(len(original)):
            if original[i] not in adj:
                adj[original[i]] = {}
            adj[original[i]][changed[i]] = cost[i]

        def dijkstra(start_node):
            dist = {chr(i): float('inf') for i in range(ord('a'), ord('z') + 1)}
            dist[start_node] = 0
            visited = set()

            while len(visited) < 26:
                min_dist = float('inf')
                min_node = None
                for node in dist:
                    if node not in visited and dist[node] < min_dist:
                        min_dist = dist[node]
                        min_node = node

                if min_node is None:
                    break

                visited.add(min_node)

                if min_node in adj:
                    for neighbor, weight in adj[min_node].items():
                        if dist[neighbor] > dist[min_node] + weight:
                            dist[neighbor] = dist[min_node] + weight
            return dist

        costs = {}
        for char in set(list(source) + list(target)):
            costs[char] = dijkstra(char)

        total_cost = 0
        for i in range(n):
            if source[i] == target[i]:
                continue
            
            if target[i] not in costs[source[i]]:
                return -1
            
            cost_to_change = costs[source[i]][target[i]]
            if cost_to_change == float('inf'):
                return -1
            total_cost += cost_to_change

        return total_cost