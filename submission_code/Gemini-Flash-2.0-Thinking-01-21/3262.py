class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        for k in range(n, 2, -1):
            sides = nums[n-k:]
            longest_side = sides[-1]
            sum_of_others = sum(sides[:-1])
            if sum_of_others > longest_side:
                return sum(sides)
        return -1