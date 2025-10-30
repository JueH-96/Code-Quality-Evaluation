class Solution:
    def hasSameDigits(self, s: str) -> bool:
        nums = [int(char) for char in s]
        while len(nums) > 2:
            next_nums = []
            for i in range(len(nums) - 1):
                total = nums[i] + nums[i+1]
                next_nums.append(total % 10)
            nums = next_nums
        return nums[0] == nums[1]