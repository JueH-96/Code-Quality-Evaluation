class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n = len(source)
        costs = {}
        for i in range(len(original)):
            if original[i] not in costs:
                costs[original[i]] = {}
            if changed[i] not in costs[original[i]] or cost[i] < costs[original[i]][changed[i]]:
                costs[original[i]][changed[i]] = cost[i]

        total_cost = 0
        for i in range(n):
            if source[i] != target[i]:
                if source[i] not in costs or target[i] not in costs[source[i]]:
                    return -1
                
                q = [(source[i], 0)]
                visited = {source[i]}
                min_cost = float('inf')
                
                while q:
                    curr, curr_cost = q.pop(0)
                    if curr == target[i]:
                        min_cost = min(min_cost, curr_cost)
                        continue
                        
                    if curr in costs:
                        for neighbor, c in costs[curr].items():
                            if neighbor not in visited:
                                visited.add(neighbor)
                                q.append((neighbor, curr_cost + c))
                
                if min_cost == float('inf'):
                    return -1
                total_cost += min_cost
        return total_cost