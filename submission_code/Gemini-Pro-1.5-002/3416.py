class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i + 1, n):
                s1 = str(nums[i])
                s2 = str(nums[j])
                diff = 0
                for k in range(len(s1)):
                    if s1[k] != s2[k]:
                        diff += 1
                ans += diff
        return ans