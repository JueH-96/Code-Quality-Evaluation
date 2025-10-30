class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        # We must form exactly 3 contiguous subarrays covering the entire array:
        #   - The first subarray starts at index 0 and ends at some index i-1.
        #   - The second subarray starts at index i and ends at some index j-1 (j > i).
        #   - The third subarray starts at index j and goes to the end (j < n).
        # The cost of each subarray is just its first element (nums[start_index]).
        # We want to minimize nums[0] + nums[i] + nums[j].
        
        min_cost = float('inf')
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                cost = nums[0] + nums[i] + nums[j]
                min_cost = min(min_cost, cost)
        return min_cost