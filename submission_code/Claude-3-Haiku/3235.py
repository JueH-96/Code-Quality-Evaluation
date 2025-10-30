class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n = len(source)
        if n != len(target):
            return -1
        
        # Create a dictionary to store the cost of changing each character
        change_cost = {}
        for i in range(len(original)):
            change_cost[(original[i], changed[i])] = cost[i]
        
        # Initialize the minimum cost to infinity
        min_cost = float('inf')
        
        # Iterate through the characters in the source string
        for i in range(n):
            if source[i] != target[i]:
                # Find the minimum cost to change the current character
                min_change_cost = float('inf')
                for j in range(len(original)):
                    if original[j] == source[i] and changed[j] == target[i]:
                        min_change_cost = min(min_change_cost, change_cost[(original[j], changed[j])])
                
                # If no valid change is found, return -1
                if min_change_cost == float('inf'):
                    return -1
                
                # Update the minimum cost
                min_cost += min_change_cost
        
        return min_cost