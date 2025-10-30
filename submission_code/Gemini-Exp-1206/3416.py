class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        s_nums = [str(num) for num in nums]
        num_digits = len(s_nums[0])

        for i in range(n):
            for j in range(i + 1, n):
                diff = 0
                for k in range(num_digits):
                    if s_nums[i][k] != s_nums[j][k]:
                        diff += 1
                ans += diff
        
        return ans * 2