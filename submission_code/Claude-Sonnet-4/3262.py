class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        
        # Calculate prefix sums for efficient sum calculation
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        # Try each position as the longest side, starting from the end
        for i in range(n - 1, 1, -1):  # Need at least 3 sides, so start from index 2
            # Sum of all sides except the longest (nums[i])
            sum_others = prefix_sum[i]
            longest_side = nums[i]
            
            # Check if we can form a valid polygon
            if sum_others > longest_side:
                # Total perimeter is sum of all sides including the longest
                return sum_others + longest_side
        
        return -1