class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        answer = [0] * n
        
        # Initialize the answer array with the direct swap costs
        for i in range(n):
            answer[i] = cost[i]
        
        # Compute the minimum cost to reach each position
        for i in range(1, n):
            # Update the cost to reach position i using the minimum of previous costs
            for j in range(i):
                answer[i] = min(answer[i], answer[j])
        
        return answer