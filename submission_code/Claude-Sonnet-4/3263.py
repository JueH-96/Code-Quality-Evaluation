class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        # Try all possible ways to split into 3 subarrays
        # First split after position i, second split after position j
        for i in range(n - 2):  # i can be 0 to n-3
            for j in range(i + 1, n - 1):  # j can be i+1 to n-2
                # Three subarrays: [0:i+1], [i+1:j+1], [j+1:n]
                # Costs are: nums[0], nums[i+1], nums[j+1]
                cost = nums[0] + nums[i + 1] + nums[j + 1]
                min_cost = min(min_cost, cost)
        
        return min_cost