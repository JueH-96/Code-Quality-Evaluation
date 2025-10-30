class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort(reverse=True)
        total = sum(nums)
        n = len(nums)
        
        for i in range(n):
            if i >= n - 2:
                return -1
            if nums[i] < total - nums[i]:
                return total
            total -= nums[i]
        
        return -1