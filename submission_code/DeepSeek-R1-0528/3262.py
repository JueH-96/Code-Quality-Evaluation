class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        prefix = 0
        ans = -1
        for num in nums:
            if prefix > num:
                ans = prefix + num
            prefix += num
        return ans