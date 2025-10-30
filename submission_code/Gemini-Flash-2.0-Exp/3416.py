class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        total_diff = 0
        for i in range(n):
            for j in range(i + 1, n):
                num1 = str(nums[i])
                num2 = str(nums[j])
                diff = 0
                for k in range(len(num1)):
                    if num1[k] != num2[k]:
                        diff += 1
                total_diff += diff
        return total_diff