class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        total_diff = 0
        for i in range(n):
            for j in range(i + 1, n):
                num1_str = str(nums[i])
                num2_str = str(nums[j])
                pair_diff = 0
                for k in range(len(num1_str)):
                    if num1_str[k] != num2_str[k]:
                        pair_diff += 1
                total_diff += pair_diff
        return total_diff