class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        sum_so_far = 0
        max_peri = -1
        
        for i in range(len(nums)):
            sum_so_far += nums[i]
            if i >= 2:
                # Check if sum of first (i) elements - nums[i] > nums[i]
                # Which is equivalent to sum_so_far > 2 * nums[i]
                if (sum_so_far - nums[i]) > nums[i]:
                    max_peri = sum_so_far
        
        return max_peri if max_peri != -1 else -1