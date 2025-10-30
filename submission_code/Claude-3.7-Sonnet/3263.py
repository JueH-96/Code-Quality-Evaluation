class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        min_cost = float('inf')
        
        # Try all possible ways to divide the array into 3 subarrays
        for i in range(1, n - 1):  # Starting index of the second subarray
            for j in range(i + 1, n):  # Starting index of the third subarray
                # Calculate the cost of each subarray
                cost1 = nums[0]  # Cost of first subarray [0...i-1]
                cost2 = nums[i]  # Cost of second subarray [i...j-1]
                cost3 = nums[j]  # Cost of third subarray [j...n-1]
                
                total_cost = cost1 + cost2 + cost3
                min_cost = min(min_cost, total_cost)
        
        return min_cost