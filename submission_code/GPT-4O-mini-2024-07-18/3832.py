from typing import List

class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        answer = [0] * n
        
        # The cost to reach position i is the minimum of the cost to swap with person i
        # or the minimum cost to reach the previous position
        for i in range(n):
            if i == 0:
                answer[i] = cost[i]
            else:
                answer[i] = min(answer[i - 1], cost[i])
        
        return answer