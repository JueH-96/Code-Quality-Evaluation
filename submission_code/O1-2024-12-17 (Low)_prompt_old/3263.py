class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # We'll choose two indices i and j (with 1 <= i < j < n) to split nums into 3 subarrays:
        # 1) nums[0:i], 2) nums[i:j], 3) nums[j:n].
        # The cost of these three subarrays is nums[0] (first element of subarray 1) 
        # + nums[i] (first element of subarray 2) + nums[j] (first element of subarray 3).
        # We want the minimum possible sum of these three values.

        min_cost = float('inf')
        for i in range(1, n-1):
            for j in range(i+1, n):
                cost = nums[0] + nums[i] + nums[j]
                min_cost = min(min_cost, cost)

        return min_cost