class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        from collections import defaultdict
        from heapq import heappop, heappush
        
        # Create a dictionary to store the minimum cost to append each character
        min_cost = defaultdict(lambda: float('inf'))
        min_cost[''] = 0
        
        # Create a min-heap to store the current state (cost, current string)
        heap = [(0, '')]
        
        while heap:
            current_cost, current_string = heappop(heap)
            
            # If the current string matches the target, return the cost
            if current_string == target:
                return current_cost
            
            # Try appending each word to the current string
            for word, cost in zip(words, costs):
                new_string = current_string + word
                new_cost = current_cost + cost
                
                # If the new string is not already in the dictionary or the new cost is lower, update the dictionary and push the new state to the heap
                if new_string not in min_cost or new_cost < min_cost[new_string]:
                    min_cost[new_string] = new_cost
                    heappush(heap, (new_cost, new_string))
        
        # If it's not possible to make the target string, return -1
        return -1