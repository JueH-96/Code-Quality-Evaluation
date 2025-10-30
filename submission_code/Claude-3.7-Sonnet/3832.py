class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        answer = [0] * n
        
        # Base case: cost to reach position 0
        answer[0] = cost[0]
        min_cost_so_far = answer[0]
        
        for i in range(1, n):
            # Either pay directly or use the minimum cost to any previous position
            answer[i] = min(cost[i], min_cost_so_far)
            # Update the minimum cost seen so far
            min_cost_so_far = min(min_cost_so_far, answer[i])
        
        return answer