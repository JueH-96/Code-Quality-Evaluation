class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
        
        num_digits = len(str(nums[0]))
        total_diff = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = 0
                num1_str = str(nums[i])
                num2_str = str(nums[j])
                for k in range(num_digits):
                    if num1_str[k] != num2_str[k]:
                        diff += 1
                total_diff += diff
        
        return total_diff