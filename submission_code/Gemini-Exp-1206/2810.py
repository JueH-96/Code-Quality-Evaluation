class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_costs = [[nums[j] for j in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                for k in range(1, n):
                    min_costs[i][j] = min(min_costs[i][j], nums[(j - k + n) % n])

        ans = float('inf')
        for ops in range(n):
            cost = ops * x
            for i in range(n):
                cost += min_costs[i][(i + ops) % n]
            ans = min(ans, cost)

        return ans