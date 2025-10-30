class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        answer = [float('inf')] * n
        
        # Start from the end and move backwards
        for i in range(n - 1, -1, -1):
            # If we are at the last person, the cost to reach them is their own cost
            if i == n - 1:
                answer[i] = cost[i]
            else:
                # Calculate the minimum cost to reach position i
                # Either we pay the cost[i] to swap directly to i
                # Or we reach i by swapping for free from a position j > i
                answer[i] = min(cost[i], answer[i + 1])
        
        return answer