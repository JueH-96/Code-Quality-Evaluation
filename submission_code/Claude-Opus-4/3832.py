class Solution:
    def minCosts(self, cost: List[int]) -> List[int]:
        n = len(cost)
        answer = [0] * n
        
        # Start from the last position (n-1) and work backwards
        min_cost_so_far = float('inf')
        
        for i in range(n - 1, -1, -1):
            # The cost to reach position i is the minimum of:
            # 1. Directly swapping with person i (cost[i])
            # 2. Going to any position j > i first (min_cost_so_far)
            min_cost_so_far = min(min_cost_so_far, cost[i])
            answer[i] = min_cost_so_far
        
        return answer