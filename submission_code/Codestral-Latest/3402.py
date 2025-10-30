class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7

        # Sort the nums array to facilitate the calculation
        nums.sort()

        # Calculate the total cost if we make all elements equal to the smallest element
        min_cost = 0
        for num in nums:
            min_cost += abs(num - nums[0]) * cost1
            min_cost %= MOD

        # Calculate the total cost if we make all elements equal to the largest element
        max_cost = 0
        for num in nums:
            max_cost += abs(num - nums[-1]) * cost1
            max_cost %= MOD

        # Calculate the total cost if we make all elements equal to any element in between
        for i in range(1, len(nums)):
            cost = 0
            for j in range(len(nums)):
                if j < i:
                    cost += (nums[i] - nums[j]) * cost1
                else:
                    cost += (nums[j] - nums[i]) * cost1
                cost %= MOD
            min_cost = min(min_cost, cost)

        return min_cost % MOD