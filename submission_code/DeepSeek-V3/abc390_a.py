class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        answer = [0] * n
        min_so_far = float('inf')
        
        for i in range(n):
            if cost[i] < min_so_far:
                min_so_far = cost[i]
            answer[i] = min_so_far
        
        return answer