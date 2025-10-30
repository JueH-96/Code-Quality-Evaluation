class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        prefix_min = [0] * n
        suffix_min = [0] * n

        # Calculate prefix_min array
        prefix_min[0] = nums[0]
        for i in range(1, n):
            prefix_min[i] = min(prefix_min[i - 1], nums[i])

        # Calculate suffix_min array
        suffix_min[n - 1] = nums[n - 1]
        for i in range(n - 2, -1, -1):
            suffix_min[i] = min(suffix_min[i + 1], nums[i])

        # Calculate the minimum cost
        min_cost = float('inf')
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cost = prefix_min[i - 1] + nums[i] + suffix_min[j]
                min_cost = min(min_cost, cost)

        return min_cost