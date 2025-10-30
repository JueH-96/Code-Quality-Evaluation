class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        answer = []
        min_cost = float('inf')
        
        for i in range(n):
            min_cost = min(min_cost, cost[i])
            answer.append(min_cost)
        
        return answer