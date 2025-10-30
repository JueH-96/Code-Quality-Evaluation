class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        
        # Calculate prefix sums
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
            
        # Check from largest possible number of sides down
        for i in range(n - 1, 1, -1):
            # For each position i, check if sum of all previous numbers > nums[i]
            if prefix_sum[i] > nums[i]:
                return prefix_sum[i + 1]
                
        return -1