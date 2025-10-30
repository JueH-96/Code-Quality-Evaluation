class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        max_num = max(nums)

        def calculate_cost(target):
            increments = [target - num for num in nums]
            total_increments = sum(increments)

            cost_only_op1 = total_increments * cost1

            num_op2 = total_increments // 2
            cost_with_op2 = num_op2 * cost2 + (total_increments - 2 * num_op2) * cost1

            return cost_with_op2

        min_cost = float('inf')
        min_cost = calculate_cost(max_num)

        return min_cost % mod