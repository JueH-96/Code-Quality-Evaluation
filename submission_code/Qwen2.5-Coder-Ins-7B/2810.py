class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        total_cost = sum(nums)
        min_cost = total_cost
        
        for i in range(1, n):
            total_cost += x
            for j in range(n):
                nums[j] = min(nums[j], nums[(j + i) % n])
                total_cost -= nums[j]
            min_cost = min(min_cost, total_cost)
        
        return min_cost