class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        # Calculate the cost for the entire array without splitting
        total_cost = 0
        sign = 1
        for num in nums:
            total_cost += num * sign
            sign *= -1
        
        # Now, find the best split to maximize the total cost
        # We need to find the best point to split the array
        # The idea is to find the maximum sum of the prefix and suffix costs
        # where the prefix is the cost of the first part and the suffix is the cost of the second part
        
        # Precompute prefix costs
        prefix = [0] * n
        sign = 1
        for i in range(n):
            prefix[i] = prefix[i-1] + nums[i] * sign if i > 0 else nums[i] * sign
            sign *= -1
        
        # Precompute suffix costs
        suffix = [0] * n
        sign = 1 if n % 2 == 1 else -1
        for i in range(n-1, -1, -1):
            suffix[i] = suffix[i+1] + nums[i] * sign if i < n-1 else nums[i] * sign
            sign *= -1
        
        # Now, find the maximum sum of prefix[i] + suffix[i+1] for all i < n-1
        max_total = total_cost
        for i in range(n-1):
            current = prefix[i] + suffix[i+1]
            if current > max_total:
                max_total = current
        
        return max_total