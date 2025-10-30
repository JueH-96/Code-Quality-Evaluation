class Solution:
    def minimumCost(self, source: str, target: str, original: list[str], changed: list[str], cost: list[int]) -> int:
        n = len(source)
        if n != len(target):
            return -1

        costs = {}
        for i in range(len(original)):
            if (original[i], changed[i]) not in costs:
                costs[(original[i], changed[i])] = cost[i]
            else:
                costs[(original[i], changed[i])] = min(costs[(original[i], changed[i])], cost[i])

        graph = {}
        for o, c in costs:
            if o not in graph:
                graph[o] = {}
            graph[o][c] = costs[(o,c)]

        total_cost = 0
        for i in range(n):
            if source[i] == target[i]:
                continue
            
            q = [(source[i], 0)]
            visited = {source[i]}
            min_cost_to_target = float('inf')
            
            while q:
                curr, curr_cost = q.pop(0)
                if curr == target[i]:
                    min_cost_to_target = curr_cost
                    break
                
                if curr in graph:
                    for neighbor, weight in graph[curr].items():
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, curr_cost + weight))
            
            if min_cost_to_target == float('inf'):
                return -1
            total_cost += min_cost_to_target

        return total_cost