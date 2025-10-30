class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # If there's only one element, the cost is simply the element itself
        if n == 1:
            return nums[0]
        
        # Calculate the cost of the entire array without splitting
        total_cost = 0
        for i in range(n):
            total_cost += nums[i] * ((-1) ** i)
        
        # We will try to maximize the cost by considering possible splits
        max_cost = total_cost
        current_cost = 0
        
        # Traverse the array and calculate the cost of subarrays
        for i in range(n - 1):
            # Calculate the cost of the current subarray ending at i
            current_cost += nums[i] * ((-1) ** i)
            
            # Calculate the cost of the remaining subarray starting at i+1
            remaining_cost = 0
            for j in range(i + 1, n):
                remaining_cost += nums[j] * ((-1) ** (j - i - 1))
            
            # Update the maximum cost if the split gives a better result
            max_cost = max(max_cost, current_cost + remaining_cost)
        
        return max_cost