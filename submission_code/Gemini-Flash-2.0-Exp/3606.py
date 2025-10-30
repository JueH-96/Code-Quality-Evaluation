class Solution:
    def minElement(self, nums: List[int]) -> int:
        new_nums = []
        for num in nums:
            s = 0
            for digit in str(num):
                s += int(digit)
            new_nums.append(s)
        return min(new_nums)