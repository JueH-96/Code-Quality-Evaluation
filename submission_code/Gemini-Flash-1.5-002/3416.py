class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        total_diff = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = 0
                num1_str = str(nums[i])
                num2_str = str(nums[j])
                for k in range(len(num1_str)):
                    if num1_str[k] != num2_str[k]:
                        diff += 1
                total_diff += diff
        return total_diff