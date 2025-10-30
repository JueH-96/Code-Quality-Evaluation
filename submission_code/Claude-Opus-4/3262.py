class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        
        # Start with all elements
        total_sum = sum(nums)
        
        # Try removing the largest elements one by one
        for i in range(len(nums) - 1, 1, -1):
            # Check if current set can form a polygon
            # Sum of all sides except the largest should be > largest side
            if total_sum - nums[i] > nums[i]:
                return total_sum
            
            # Remove the current largest element for next iteration
            total_sum -= nums[i]
        
        return -1