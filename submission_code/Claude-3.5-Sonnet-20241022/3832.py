class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        # Initialize answer array with infinity
        answer = [float('inf')] * n
        
        # For each target position
        for target in range(n):
            # Try reaching target from each position in front of it
            for start in range(target + 1):
                # If we're starting from position 'start',
                # we need to pay cost[start] to swap with that person
                current_cost = cost[start]
                
                # If we're not directly swapping to target position,
                # we can get there for free after reaching 'start'
                if start < target:
                    # We only need to pay the minimum cost to reach any position
                    # from start to target, as we can move backwards for free
                    if start > 0:
                        current_cost = min(current_cost, answer[start - 1])
                
                # Update the minimum cost to reach target
                answer[target] = min(answer[target], current_cost)
        
        return answer