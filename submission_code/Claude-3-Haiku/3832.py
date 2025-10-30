class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        answer = [0] * n
        
        # Initialize the last position
        answer[n-1] = cost[n-1]
        
        # Iterate from the second-to-last position to the first position
        for i in range(n-2, -1, -1):
            # If the person in front of us is behind us, we can swap for free
            if i < n-1:
                answer[i] = min(answer[i+1], cost[i] + answer[i+1])
            # If the person in front of us is in front of us, we have to pay the cost
            else:
                answer[i] = cost[i]
        
        return answer