class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        for i in range(n - 1, 1, -1):
            longest_side = nums[i]
            other_sides = nums[:i]
            sum_other_sides = sum(other_sides)
            if sum_other_sides > longest_side:
                return sum(nums[:i+1])
        return -1