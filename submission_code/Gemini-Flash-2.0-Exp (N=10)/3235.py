import heapq

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        if len(source) != len(target):
            return -1

        n = len(source)
        adj = {}
        for i in range(len(original)):
            if original[i] not in adj:
                adj[original[i]] = {}
            adj[original[i]][changed[i]] = cost[i]

        total_cost = 0
        for i in range(n):
            if source[i] == target[i]:
                continue
            
            start_char = source[i]
            end_char = target[i]
            
            if start_char not in adj:
                return -1
            
            min_cost = float('inf')
            
            pq = [(0, start_char)]
            visited = set()
            
            while pq:
                curr_cost, curr_char = heapq.heappop(pq)
                
                if curr_char == end_char:
                    min_cost = curr_cost
                    break
                
                if (curr_char) in visited:
                    continue
                visited.add(curr_char)
                
                if curr_char in adj:
                    for neighbor, cost_to_neighbor in adj[curr_char].items():
                        heapq.heappush(pq, (curr_cost + cost_to_neighbor, neighbor))
            
            if min_cost == float('inf'):
                return -1
            
            total_cost += min_cost
            
        return total_cost