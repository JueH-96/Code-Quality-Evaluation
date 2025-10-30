class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        ans = -1
        n = len(nums)
        for i in range(n - 1):
            if nums[i+1] - nums[i] == 1:
                length = 2
                diff = -1
                for j in range(i + 2, n):
                    if nums[j] - nums[j-1] == diff:
                        length += 1
                        diff *= -1
                    else:
                        break
                ans = max(ans, length)
        return ans