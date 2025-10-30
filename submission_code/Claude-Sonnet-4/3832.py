class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        answer = [0] * n
        
        # For position 0, we must pay cost[0]
        answer[0] = cost[0]
        
        # For each subsequent position, we take the minimum cost
        # seen so far (since we can reach any previous position
        # and then move forward for free)
        for i in range(1, n):
            answer[i] = min(answer[i-1], cost[i])
        
        return answer