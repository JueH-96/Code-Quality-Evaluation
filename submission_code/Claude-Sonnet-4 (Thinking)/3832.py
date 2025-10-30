class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        result = [0] * n
        
        result[0] = cost[0]
        
        for i in range(1, n):
            result[i] = min(cost[i], result[i-1])
        
        return result