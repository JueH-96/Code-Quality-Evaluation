class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        total = sum(nums)
        
        # Start from the end and work backwards
        for i in range(len(nums) - 1, 1, -1):
            # Check if elements from index 0 to i can form a polygon
            # For a valid polygon: sum of all sides > 2 * largest side
            if total > 2 * nums[i]:
                return total
            # Remove the largest element from consideration
            total -= nums[i]
        
        return -1