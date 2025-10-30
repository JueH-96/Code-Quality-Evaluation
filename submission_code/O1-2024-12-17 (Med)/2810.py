class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        
        # cost[t][d] = cost of buying a chocolate of type t at iteration d
        # which is simply nums[(t - d) % n], because index i with cost nums[i]
        # becomes type (i + d) % n. So to get type t at iteration d, i = (t - d) % n.
        cost = [[0]*n for _ in range(n)]
        for t in range(n):
            for d in range(n):
                cost[t][d] = nums[(t - d) % n]
        
        # minCost[t][d] = minimum cost of buying type t in any iteration from 0..d
        minCost = [[0]*n for _ in range(n)]
        for t in range(n):
            minCost[t][0] = cost[t][0]
            for d in range(1, n):
                minCost[t][d] = min(minCost[t][d-1], cost[t][d])
        
        # Try all possible numbers of operations m from 0..(n-1).
        # If we perform m operations, then our total cost is:
        #   (m * x) + sum_{t=0..n-1}( minimum cost to buy type t among iterations 0..m )
        ans = float('inf')
        for m in range(n):
            # sum of picking each type t from its cheapest iteration in [0..m]
            chocolate_cost = sum(minCost[t][m] for t in range(n))
            ans = min(ans, chocolate_cost + m * x)
        
        return ans