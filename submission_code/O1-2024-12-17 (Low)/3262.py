class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        # Sort the array in non-decreasing order
        nums.sort()
        
        # Calculate the total sum of all elements
        total_sum = sum(nums)
        
        # Start from the largest index and keep removing elements
        # if they fail the polygon condition
        i = len(nums) - 1
        while i >= 2:
            # The largest side is nums[i]. Check if sum of all sides
            # is more than twice this side (polygon condition).
            if total_sum > 2 * nums[i]:
                return total_sum
            # Otherwise, remove this side from the sum and
            # move to the next largest side
            total_sum -= nums[i]
            i -= 1
        
        # If no valid polygon is found, return -1
        return -1