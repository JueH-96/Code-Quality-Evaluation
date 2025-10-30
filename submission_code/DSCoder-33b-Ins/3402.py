class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        def calculate_cost(target):
            cost = 0
            for i in range(len(nums)):
                if nums[i] < target:
                    cost += (target - nums[i]) * cost1
                elif nums[i] > target:
                    cost += (nums[i] - target) * cost2
            return cost

        left, right = min(nums), max(nums)
        while left < right:
            mid = (left + right) // 2
            if calculate_cost(mid) < calculate_cost(mid + 1):
                right = mid
            else:
                left = mid + 1

        return calculate_cost(left) % (10**9 + 7)