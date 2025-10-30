class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        from collections import defaultdict
        import heapq
        
        # Create a graph of transformations with costs
        graph = defaultdict(dict)
        for o, c, co in zip(original, changed, cost):
            if c in graph[o]:
                graph[o][c] = min(graph[o][c], co)  # Use the minimum cost for transformation
            else:
                graph[o][c] = co
        
        # Use Dijkstra's algorithm to find the minimum cost to transform each character to every other character
        def dijkstra(source_char):
            # Min-heap priority queue
            heap = [(0, source_char)]  # (cost, character)
            min_cost = {}
            while heap:
                current_cost, current_char = heapq.heappop(heap)
                
                if current_char in min_cost:
                    continue
                
                min_cost[current_char] = current_cost
                
                for neighbor in graph[current_char]:
                    if neighbor not in min_cost:
                        heapq.heappush(heap, (current_cost + graph[current_char][neighbor], neighbor))
            
            return min_cost
        
        # Precompute the minimum costs to transform each character to any other character
        all_costs = {char: dijkstra(char) for char in graph}
        
        total_cost = 0
        
        # Calculate the total minimum cost to transform source to target
        for s_char, t_char in zip(source, target):
            if s_char == t_char:
                continue  # No cost if characters are already the same
            
            if t_char not in all_costs.get(s_char, {}):
                return -1  # If there's no way to transform s_char to t_char
            
            total_cost += all_costs[s_char][t_char]
        
        return total_cost