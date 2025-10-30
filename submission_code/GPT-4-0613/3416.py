class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        total = 0
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                total += self.digitDifference(nums[i], nums[j])
        return total

    def digitDifference(self, num1: int, num2: int) -> int:
        diff = 0
        while num1 > 0 or num2 > 0:
            diff += abs((num1 % 10) - (num2 % 10))
            num1 //= 10
            num2 //= 10
        return diff