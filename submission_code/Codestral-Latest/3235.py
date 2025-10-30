class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        from collections import defaultdict
        import heapq

        # Create a graph where each node represents a character
        graph = defaultdict(list)

        # Build the graph based on the given transformations
        for i in range(len(original)):
            graph[original[i]].append((changed[i], cost[i]))

        # Initialize a priority queue for Dijkstra's algorithm
        pq = [(0, source[0])]
        # Dictionary to store the minimum cost to reach each character
        min_cost = {source[0]: 0}

        # Dijkstra's algorithm to find the minimum cost to transform each character
        while pq:
            curr_cost, curr_char = heapq.heappop(pq)

            if curr_cost > min_cost.get(curr_char, float('inf')):
                continue

            for next_char, edge_cost in graph[curr_char]:
                new_cost = curr_cost + edge_cost
                if new_cost < min_cost.get(next_char, float('inf')):
                    min_cost[next_char] = new_cost
                    heapq.heappush(pq, (new_cost, next_char))

        # Calculate the total minimum cost to transform source to target
        total_cost = 0
        for s, t in zip(source, target):
            if s != t:
                if t not in min_cost:
                    return -1
                total_cost += min_cost[t]

        return total_cost