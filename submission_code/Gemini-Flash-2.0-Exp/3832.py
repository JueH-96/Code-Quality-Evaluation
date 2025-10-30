class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        answer = [0] * n
        
        for i in range(n):
            answer[i] = cost[i]
            
        min_cost = float('inf')
        for c in cost:
            min_cost = min(min_cost, c)
            
        for i in range(n):
            answer[i] = min_cost
            
        return answer