class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        for i in range(len(nums), 2, -1):
            sub_ar = nums[:i]
            if sum(sub_ar[:-1]) > sub_ar[-1] and i >= 3:
                return sum(sub_ar)
        return -1